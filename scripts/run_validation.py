"""Validate the change-point detector against real StatsBomb ground truth.

Needs PITCHGRAPH_DATA pointing at the built lake (same env var pitchgraph
itself uses). For every eligible competition-season (the same set the
Library build uses: >=4 teams with >=10 matches), and for every team's
real Tactical Shift / red card events, checks whether the detector --
seeded with that team's shrunk chain fit on every OTHER match (no
leakage) -- flags a change point within a possession window.

Thresholds (hazard, prior_strength, detection threshold) are provisional
defaults, not yet calibrated via nulls.py's permutation approach -- this
is a first empirical read on the real lake, not the final calibrated
validation.

    python scripts/run_validation.py
    python scripts/run_validation.py --keys 2_27 11_27   # restrict to some seasons
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from pitchgraph.chains.counts import CountStore  # noqa: E402
from pitchgraph.chains.grid import TACTICAL  # noqa: E402
from pitchgraph.chains.validate import prepare as prepare_league  # noqa: E402
from pitchgraph.data.lake import Lake  # noqa: E402
from pitchgraph.library.build import eligible_seasons  # noqa: E402

from dynamic_shift.ground_truth import red_cards, tactical_shifts  # noqa: E402
from dynamic_shift.validate import detect_within, leave_one_out_prior  # noqa: E402

HAZARD = 20.0
PRIOR_STRENGTH = 10.0
THRESHOLD = 0.15
WITHIN_POSSESSIONS = 15
CHANGE_POINT_WINDOW = 5


def season_events(lake: Lake, cid: int, sid: int) -> pl.LazyFrame:
    return lake.events().filter((pl.col("competition_id") == cid) & (pl.col("season_id") == sid))


def run_season(lake: Lake, cid: int, sid: int, label: str, log=print) -> dict:
    events = season_events(lake, cid, sid)
    store = CountStore.from_events(events, TACTICAL)
    league, teams, kappa, _ = prepare_league(store)

    buckets = {"real_formation_change": [], "same_formation": [], "red_card": []}
    for team_id in teams:
        tc = store.team(team_id)
        for match_id in tc.match_ids.tolist():
            match_events = events.filter(pl.col("match_id") == match_id)
            shifts = tactical_shifts(match_events, team_id)
            cards = [c for c in red_cards(match_events) if c["team_id"] == team_id]

            controls = (
                [(s["minute"], "real_formation_change") for s in shifts if s["real_change"]]
                + [(s["minute"], "same_formation") for s in shifts if not s["real_change"]]
                + [(c["minute"], "red_card") for c in cards]
            )
            if not controls:
                continue

            chain = leave_one_out_prior(store, team_id, exclude_match=match_id, kappa=kappa)
            for minute, bucket in controls:
                result = detect_within(
                    match_events, team_id, chain, around_minute=minute,
                    prior_strength=PRIOR_STRENGTH, hazard=HAZARD, threshold=THRESHOLD,
                    within_possessions=WITHIN_POSSESSIONS,
                    change_point_window=CHANGE_POINT_WINDOW, grid=TACTICAL,
                )
                buckets[bucket].append(result.latency if result.detected else None)

    summary = {}
    for bucket, results in buckets.items():
        n = len(results)
        detected = [r for r in results if r is not None]
        summary[bucket] = {
            "n": n,
            "detection_rate": round(len(detected) / n, 3) if n else None,
            "mean_latency": round(sum(detected) / len(detected), 2) if detected else None,
        }
    log(f"  {label}: " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*", help="competition-season keys, e.g. 2_27")
    args = ap.parse_args()

    lake = Lake.open()
    seasons = eligible_seasons(lake)
    if args.keys:
        seasons = seasons.filter(pl.col("key").is_in(args.keys))
    print(f"{seasons.height} eligible competition-seasons")

    t0 = time.time()
    all_results = {}
    for row in seasons.iter_rows(named=True):
        label = f"{row['competition']} {row['season']}"
        all_results[row["key"]] = run_season(lake, row["competition_id"], row["season_id"], label)

    totals = {"real_formation_change": [0, 0], "same_formation": [0, 0], "red_card": [0, 0]}
    for season_summary in all_results.values():
        for bucket, s in season_summary.items():
            totals[bucket][0] += s["n"]
            totals[bucket][1] += round((s["detection_rate"] or 0) * s["n"])

    print(f"\n== overall (hazard={HAZARD}, prior_strength={PRIOR_STRENGTH}, "
          f"threshold={THRESHOLD}, within={WITHIN_POSSESSIONS} possessions) ==")
    for bucket, (n, detected) in totals.items():
        rate = round(detected / n, 3) if n else None
        print(f"  {bucket}: n={n}, detected={detected}, rate={rate}")
    print(f"\ndone in {time.time() - t0:.0f}s")
