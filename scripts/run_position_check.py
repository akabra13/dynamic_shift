"""v2 test A: do continuing players change where they touch the ball around
a labelled formation change more than around a random moment?
Pre-registered in the README ("v2: identity-aware tests").

    python scripts/run_position_check.py                 # development seasons
    python scripts/run_position_check.py --split lockbox # final run, once
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

from pitchgraph.data.lake import Lake  # noqa: E402

from dynamic_shift.ground_truth import classify_shifts, shift_details  # noqa: E402
from dynamic_shift.positions import FEATURE_NAMES, WINDOW_SECONDS, elapsed, window_change  # noqa: E402
from dynamic_shift.report import family_report  # noqa: E402
from dynamic_shift.split import is_lockbox  # noqa: E402

RANDOM_POINTS_PER_MATCH = 5
LABELS = ("real_substantive", "real_clean", "same_formation", "random")


def shift_times(me: pl.DataFrame, team_id: int) -> dict:
    """(possession, minute+second/60) -> (period, period_seconds) of each Tactical Shift."""
    r = me.filter((pl.col("type") == "Tactical Shift") & (pl.col("team_id") == team_id))
    return {(d["possession"], round(d["minute"] + d["second"] / 60.0, 4)): (d["period"], d["period_seconds"])
            for d in r.iter_rows(named=True)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "lockbox"], default="dev")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lake = Lake.open()
    seasons = (lake.matches().select("competition_id", "season_id", "competition", "season").unique()
               .with_columns(key=pl.concat_str(["competition_id", "season_id"], separator="_"))
               .sort("competition", "season"))
    seasons = seasons.filter(pl.col("key").map_elements(is_lockbox, return_dtype=pl.Boolean)
                             == (args.split == "lockbox"))
    if args.keys:
        seasons = seasons.filter(pl.col("key").is_in(args.keys))
    print(f"{args.split}: {seasons.height} competition-seasons")

    t0 = time.time()
    fam = {n: {lab: [] for lab in LABELS} for n in FEATURE_NAMES}
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
            lazy = me.lazy()
            for team_id in me["team_id"].drop_nulls().unique().to_list():
                touches = me.with_columns(pl.Series("t", t_all)).filter(
                    (pl.col("team_id") == team_id) & pl.col("player_id").is_not_null()
                    & pl.col("x").is_not_null() & (pl.col("position") != "Goalkeeper"))
                if touches.height < 100:
                    continue
                t, p, x, y = (touches[c].to_numpy() for c in ("t", "player_id", "x", "y"))
                key = key_ids.setdefault((row["key"], match_id, team_id), len(key_ids))
                times = shift_times(me, team_id)
                moments = {lab: [] for lab in LABELS}
                for lab, lst in classify_shifts(shift_details(lazy, team_id)).items():
                    for d in lst:
                        pt = times.get((d["possession"], round(d["minute"], 4)))
                        if pt is None:
                            continue
                        moments[lab].append(float(pt[1] if pt[0] == 1 else p1_end + pt[1]))
                if end > 2 * WINDOW_SECONDS:
                    moments["random"] = rng.uniform(WINDOW_SECONDS, end - WINDOW_SECONDS,
                                                    size=RANDOM_POINTS_PER_MATCH).tolist()
                for lab, ts in moments.items():
                    for m in ts:
                        f = window_change(t, p, x, y, m, end)
                        if f:
                            for n in FEATURE_NAMES:
                                fam[n][lab].append((key, {n: f[n]}))

    print("\nusable events: " + str({n: {l: len(v) for l, v in fam[n].items()} for n in FEATURE_NAMES}))
    results = {"split": args.split, "window_seconds": WINDOW_SECONDS,
               **{n: family_report(n, fam[n], (n,), rng) for n in FEATURE_NAMES}}
    Path("position_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
