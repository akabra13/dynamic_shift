"""v2 test B: can a team's back-line size and forward count be recognised from
where its players touch the ball? Pre-registered in the README
("v2: identity-aware tests").

    python scripts/run_formation_check.py                  # cross-validated on dev seasons
    python scripts/run_formation_check.py --split lockbox  # train on dev, score lockbox once
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from dynamic_shift.formation import balanced_accuracy, parse_formation, shape_vector  # noqa: E402
from dynamic_shift.positions import MIN_TOUCHES, centroids, elapsed  # noqa: E402
from dynamic_shift.split import is_lockbox  # noqa: E402

MIN_SEGMENT_SECONDS = 600.0
TARGETS = ("back_line", "forwards")
PASS_BA = 0.60
PASS_LOWER = 0.40
N_BOOT = 500


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          class_weight="balanced", random_state=0)


def boot_ba(y, pred, groups, rng):
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq))])
        vals.append(balanced_accuracy(y[pick], pred[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def segments(me: pl.DataFrame, team_id: int, t_all: np.ndarray, end: float, p1_end: float):
    """Yield (formation dict, 20-vector) for every long-enough tactical segment."""
    tac = (me.filter(pl.col("type").is_in(["Starting XI", "Tactical Shift"]) & (pl.col("team_id") == team_id))
             .sort("idx"))
    rows = []
    for d in tac.iter_rows(named=True):
        tact = (json.loads(d["extra"]) if d["extra"] else {}).get("tactics") or {}
        form = parse_formation(tact.get("formation"))
        lineup = [p["player"]["id"] for p in tact.get("lineup") or []
                  if p.get("position", {}).get("name") != "Goalkeeper"]
        t = d["period_seconds"] if d["period"] == 1 else p1_end + d["period_seconds"]
        if d["period"] in (1, 2) and form and len(lineup) == 10:
            rows.append((t, form, lineup))
    touches = me.with_columns(pl.Series("t", t_all)).filter(
        (pl.col("team_id") == team_id) & pl.col("player_id").is_not_null() & pl.col("x").is_not_null())
    t, p, x, y = (touches[c].to_numpy() for c in ("t", "player_id", "x", "y"))
    for i, (t0, form, lineup) in enumerate(rows):
        t1 = rows[i + 1][0] if i + 1 < len(rows) else end
        if t1 - t0 < MIN_SEGMENT_SECONDS:
            continue
        cents = {k: v for k, v in centroids(t, p, x, y, t0, t1, MIN_TOUCHES).items() if k in lineup}
        vec = shape_vector(cents)
        if vec is not None:
            yield form, vec


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "lockbox"], default="dev")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lake = Lake.open()
    seasons = (lake.matches().select("competition_id", "season_id", "competition", "season").unique()
               .with_columns(key=pl.concat_str(["competition_id", "season_id"], separator="_"))
               .sort("competition", "season"))
    seasons = seasons.with_columns(box=pl.col("key").map_elements(is_lockbox, return_dtype=pl.Boolean))

    t0 = time.time()
    X, Y, G, BOX = [], {k: [] for k in TARGETS}, [], []
    for row in seasons.iter_rows(named=True):
        ev = (lake.events().filter((pl.col("competition_id") == row["competition_id"])
                                   & (pl.col("season_id") == row["season_id"]))
              .filter(pl.col("period").is_in([1, 2])).collect())
        print(f"{row['competition']} {row['season']} ({row['key']}) [{time.time() - t0:.0f}s]", flush=True)
        for (match_id,), me in ev.partition_by("match_id", as_dict=True).items():
            t_all = elapsed(me["period"].to_numpy(), me["period_seconds"].to_numpy())
            end = float(np.nanmax(t_all))
            p1 = me.filter(pl.col("period") == 1)["period_seconds"]
            p1_end = float(p1.max()) if p1.len() else 0.0
            for team_id in me["team_id"].drop_nulls().unique().to_list():
                for form, vec in segments(me, team_id, t_all, end, p1_end):
                    X.append(vec); G.append(match_id); BOX.append(row["box"])
                    for k in TARGETS:
                        Y[k].append(form[k])

    X, G, BOX = np.array(X), np.array(G), np.array(BOX)
    print(f"\nsegments: dev {int((~BOX).sum())}, lockbox {int(BOX.sum())}")
    results = {"split": args.split, "targets": {}}
    for k in TARGETS:
        y = np.array(Y[k])
        dev = ~BOX
        if args.split == "dev":
            idx, pred = np.flatnonzero(dev), np.empty(dev.sum(), dtype=y.dtype)
            for tr, te in GroupKFold(5).split(X[dev], y[dev], G[dev]):
                pred[te] = model().fit(X[dev][tr], y[dev][tr]).predict(X[dev][te])
            yt, gt = y[dev], G[dev]
        else:
            pred = model().fit(X[dev], y[dev]).predict(X[BOX])
            yt, gt = y[BOX], G[BOX]
        ba = balanced_accuracy(yt, pred)
        lo, hi = boot_ba(yt, pred, gt, rng)
        classes, counts = np.unique(yt, return_counts=True)
        results["targets"][k] = {
            "n": int(len(yt)), "balanced_accuracy": round(ba, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "chance": round(1 / len(classes), 4),
            "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
            "passes_bar": bool(ba >= PASS_BA and lo > PASS_LOWER)}
        print(f"{k:10s} n={len(yt)} BA={ba:.3f} (95% CI {lo:.3f}-{hi:.3f}) chance={1/len(classes):.3f}"
              + ("   PASSES BAR" if results["targets"][k]["passes_bar"] else ""))
    Path("formation_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
