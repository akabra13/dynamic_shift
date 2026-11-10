"""v2c: does the predicted back line change around a labelled 3<->4 switch?
Pre-registered in the README ("Pre-registered v2c").

    python scripts/run_flip_check.py                  # development seasons, team-cross-fitted
    python scripts/run_flip_check.py --split lockbox  # train on dev, score lockbox once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from dynamic_shift.formation import parse_formation, window_vector  # noqa: E402
from dynamic_shift.ground_truth import shift_details  # noqa: E402
from dynamic_shift.positions import elapsed  # noqa: E402
from dynamic_shift.signal_check import auc_ci_by_match  # noqa: E402
from dynamic_shift.split import is_lockbox  # noqa: E402

WIN = 900.0
MIN_TRAIN_SEGMENT = 1800.0
TRAIN_PER_SEGMENT = 3
RANDOM_PER_MATCH = 5
N_FOLDS = 5
LABELS = ("back_switch", "other_change", "same_formation", "random")
BAR_RANDOM, BAR_SAME = 0.70, 0.65


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          class_weight="balanced", random_state=0)


def team_fold(team_key: str) -> int:
    return int(hashlib.md5(team_key.encode()).hexdigest(), 16) % N_FOLDS


def back(code):
    f = parse_formation(code)
    return f["back_line"] if f else None


def tactical_rows(me, team_id, p1_end):
    tac = me.filter(pl.col("type").is_in(["Starting XI", "Tactical Shift"])
                    & (pl.col("team_id") == team_id)).sort("idx")
    out = []
    for d in tac.iter_rows(named=True):
        if d["period"] not in (1, 2):
            continue
        tact = (json.loads(d["extra"]) if d["extra"] else {}).get("tactics") or {}
        t = d["period_seconds"] if d["period"] == 1 else p1_end + d["period_seconds"]
        out.append((t, back(tact.get("formation"))))
    return out


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
    if args.split == "dev":
        seasons = seasons.filter(~pl.col("box"))

    t0 = time.time()
    TX, TY, TFOLD, TBOX = [], [], [], []           # training windows
    MB, MA, MLAB, MKEY, MFOLD, MBOX = [], [], [], [], [], []   # moments
    key_ids: dict = {}

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
                team_key = f"{row['key']}_{team_id}"
                fold = team_fold(team_key)
                touches = me.with_columns(pl.Series("t", t_all)).filter(
                    (pl.col("team_id") == team_id) & pl.col("player_id").is_not_null()
                    & pl.col("x").is_not_null() & (pl.col("position") != "Goalkeeper"))
                if touches.height < 100:
                    continue
                t, p, x, y = (touches[c].to_numpy() for c in ("t", "player_id", "x", "y"))

                # training windows inside long labelled segments
                rows = tactical_rows(me, team_id, p1_end)
                for i, (s0, bl) in enumerate(rows):
                    s1 = rows[i + 1][0] if i + 1 < len(rows) else end
                    if bl not in (3, 4) or s1 - s0 < MIN_TRAIN_SEGMENT:
                        continue
                    for lo in rng.uniform(s0, s1 - WIN, size=TRAIN_PER_SEGMENT):
                        v = window_vector(t, p, x, y, lo, lo + WIN)
                        if v is not None:
                            TX.append(v); TY.append(bl); TFOLD.append(fold); TBOX.append(row["box"])

                # moments
                times = {}
                for d in me.filter((pl.col("type") == "Tactical Shift") & (pl.col("team_id") == team_id)
                                   ).iter_rows(named=True):
                    times[(d["possession"], round(d["minute"] + d["second"] / 60.0, 4))] = (
                        d["period_seconds"] if d["period"] == 1 else p1_end + d["period_seconds"])
                moments = []
                for d in shift_details(me.lazy(), team_id):
                    m = times.get((d["possession"], round(d["minute"], 4)))
                    if m is None:
                        continue
                    if not d["real_change"]:
                        lab = "same_formation"
                    else:
                        b0, b1 = back(d["formation_before"]), back(d["formation_after"])
                        lab = "back_switch" if {b0, b1} == {3, 4} else "other_change"
                    moments.append((lab, m))
                if end > 2 * WIN:
                    moments += [("random", m) for m in rng.uniform(WIN, end - WIN, size=RANDOM_PER_MATCH)]
                key = key_ids.setdefault((row["key"], match_id, team_id), len(key_ids))
                for lab, m in moments:
                    if m - WIN < 0 or m + WIN > end:
                        continue
                    vb = window_vector(t, p, x, y, m - WIN, m)
                    va = window_vector(t, p, x, y, m, m + WIN)
                    if vb is not None and va is not None:
                        MB.append(vb); MA.append(va); MLAB.append(lab); MKEY.append(key)
                        MFOLD.append(fold); MBOX.append(row["box"])

    TX, TY, TFOLD, TBOX = map(np.array, (TX, TY, TFOLD, TBOX))
    MB, MA, MLAB, MKEY, MFOLD, MBOX = map(np.array, (MB, MA, MLAB, MKEY, MFOLD, MBOX))
    print(f"\ntraining windows {len(TX)} (back3 {int((TY == 3).sum())}), moments {len(MLAB)}: "
          + str({l: int((MLAB == l).sum()) for l in LABELS}))

    p_before, p_after = np.full(len(MLAB), np.nan), np.full(len(MLAB), np.nan)
    if args.split == "dev":
        for k in range(N_FOLDS):
            clf = model().fit(TX[TFOLD != k], TY[TFOLD != k])
            m = MFOLD == k
            if m.any():
                col = list(clf.classes_).index(3)
                p_before[m] = clf.predict_proba(MB[m])[:, col]
                p_after[m] = clf.predict_proba(MA[m])[:, col]
        score_mask = np.ones(len(MLAB), bool)
    else:
        clf = model().fit(TX[~TBOX], TY[~TBOX])
        score_mask = MBOX
        col = list(clf.classes_).index(3)
        p_before[score_mask] = clf.predict_proba(MB[score_mask])[:, col]
        p_after[score_mask] = clf.predict_proba(MA[score_mask])[:, col]

    flip = np.abs(p_after - p_before)
    lab, keys, flip = MLAB[score_mask], MKEY[score_mask], flip[score_mask]
    pb, pa = p_before[score_mask], p_after[score_mask]

    def auc(a, b):
        r = auc_ci_by_match(flip[lab == a], keys[lab == a], flip[lab == b], keys[lab == b], rng, level=0.95)
        return {"auc": round(r[0], 4), "ci95": [round(r[1], 4), round(r[2], 4)]}

    res = {"split": args.split, "window_seconds": WIN,
           "n": {l: int((lab == l).sum()) for l in LABELS},
           "mean_flip": {l: round(float(flip[lab == l].mean()), 4) for l in LABELS if (lab == l).any()},
           "vs_random": {l: auc(l, "random") for l in LABELS if l != "random"},
           "switch_vs_same_formation": auc("back_switch", "same_formation")}
    sw = lab == "back_switch"
    crossed = ((pb[sw] < 0.5) & (pa[sw] >= 0.5)) | ((pb[sw] >= 0.5) & (pa[sw] < 0.5))
    res["switch_crosses_half"] = round(float(crossed.mean()), 4) if sw.any() else None
    r1, r2 = res["vs_random"]["back_switch"], res["switch_vs_same_formation"]
    res["passes_bar"] = bool(r1["auc"] >= BAR_RANDOM and r1["ci95"][0] > 0.5
                             and r2["auc"] >= BAR_SAME and r2["ci95"][0] > 0.5)
    print(json.dumps(res, indent=1))
    Path("flip_check_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
