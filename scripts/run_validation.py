"""Validate the change-point detector against real StatsBomb ground truth.

Needs PITCHGRAPH_DATA pointing at the built lake (same env var pitchgraph
itself uses). For every eligible competition-season (the same set the
Library build uses: >=4 teams with >=10 matches), and for every team's
matches:

  Pass 1: score the real possession sequence once, and score
  `N_PERMUTATIONS` within-half shuffles of it (the permutation null).
  Pool every null trajectory across the whole run and calibrate ONE
  global alarm threshold from it (nulls.calibrate_threshold), targeting
  `FALSE_ALARM_TARGET` false alarms per team-match. This is label-free:
  it never looks at a single ground-truth event, so evaluating on the
  same matches afterwards doesn't leak labels into calibration.

  Pass 2: turn each real trajectory into a list of alarms at that
  threshold, and check four control classes against them: real
  formation changes, same-formation shifts (negative control), red
  cards, and random time points (the base rate -- what "detected"
  means for a point with no claimed significance at all).

Hyperparameters (hazard, prior_strength, burn-in, refractory, window)
are fixed before looking at any result, not tuned on these labels.

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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pitchgraph.analysis.reliability import estimate_ratio  # noqa: E402
from pitchgraph.chains.counts import CountStore  # noqa: E402
from pitchgraph.chains.grid import TACTICAL  # noqa: E402
from pitchgraph.chains.validate import prepare as prepare_league  # noqa: E402
from pitchgraph.data.lake import Lake  # noqa: E402
from pitchgraph.library.build import eligible_seasons  # noqa: E402

from dynamic_shift.ground_truth import had_substitution_within, red_cards, tactical_shifts  # noqa: E402
from dynamic_shift.nulls import calibrate_threshold, false_alarm_rate, permutation_null_scores  # noqa: E402
from dynamic_shift.observe import chain_prior_alpha, match_possession_counts  # noqa: E402
from dynamic_shift.validate import alarms, check_control, leave_one_out_prior, score_match  # noqa: E402

HAZARD = 100.0
PRIOR_STRENGTH = 10.0
CHANGE_POINT_WINDOW = 5
BURN_IN = 10
REFRACTORY = 5
WITHIN_POSSESSIONS = 15
N_PERMUTATIONS = 20
FALSE_ALARM_TARGET = 0.25
RANDOM_POINTS_PER_MATCH = 2
MIN_POSSESSIONS = BURN_IN + 5


def collect_team_matches(lake: Lake, cid: int, sid: int, rng: np.random.Generator,
                          log=print) -> list[dict]:
    """One entry per team-match: the real score trajectory, its null
    trajectories, and every control event to check against them.

    Events are collected once per competition-season and partitioned by
    match, rather than re-filtering the season's lazy frame on every
    team-match -- with ~2 teams and several control events per match,
    that was re-scanning the same data many times over.
    """
    events_df = (lake.events()
                 .filter((pl.col("competition_id") == cid) & (pl.col("season_id") == sid))
                 .collect())
    store = CountStore.from_events(events_df.lazy(), TACTICAL)
    league, teams, kappa, _ = prepare_league(store)
    by_match = {k[0]: df.lazy() for k, df in events_df.partition_by("match_id", as_dict=True).items()}

    out = []
    for team_id in teams:
        tc = store.team(team_id)
        for match_id in tc.match_ids.tolist():
            match_events = by_match.get(match_id)
            if match_events is None:
                continue

            possessions = match_possession_counts(match_events, team_id, TACTICAL)
            if len(possessions) < MIN_POSSESSIONS:
                continue

            chain = leave_one_out_prior(store, team_id, exclude_match=match_id, kappa=kappa)
            alpha = chain_prior_alpha(chain, kappa=PRIOR_STRENGTH)

            real_scores = score_match(match_events, team_id, chain, PRIOR_STRENGTH, HAZARD,
                                       CHANGE_POINT_WINDOW, TACTICAL)
            null_scores = permutation_null_scores(possessions, alpha, HAZARD, CHANGE_POINT_WINDOW,
                                                   N_PERMUTATIONS, rng)

            shifts = tactical_shifts(match_events, team_id)
            cards = [c for c in red_cards(match_events) if c["team_id"] == team_id]
            controls = (
                [{"possession": s["possession"], "label": "real_formation_change",
                  "near_sub": had_substitution_within(match_events, team_id, s["minute"])}
                 for s in shifts if s["real_change"]]
                + [{"possession": s["possession"], "label": "same_formation"}
                   for s in shifts if not s["real_change"]]
                + [{"possession": c["possession"], "label": "red_card"} for c in cards]
            )

            eligible_random = possessions[BURN_IN:]
            k = min(RANDOM_POINTS_PER_MATCH, len(eligible_random))
            if k:
                picks = rng.choice(len(eligible_random), size=k, replace=False)
                controls += [{"possession": eligible_random[p][0], "label": "random"} for p in picks]

            out.append({"competition_id": cid, "season_id": sid, "team_id": team_id,
                        "match_id": match_id, "real_scores": real_scores,
                        "null_scores": null_scores, "controls": controls})
    log(f"  {len(teams)} teams, {len(out)} team-matches")
    return out


def summarise(rows_by_label: dict, latencies_by_label: dict) -> dict:
    """Bootstrap-by-match detection rate per label: one row per team-match
    (num detected / num controls in that match), not one row per control
    event, so a match with several same-formation shifts doesn't get
    resampled as if those were independent matches."""
    summary = {}
    for label, by_match in rows_by_label.items():
        num = np.array([d for d, _ in by_match.values()], dtype=float)
        den = np.array([n for _, n in by_match.values()], dtype=float)
        est = estimate_ratio(num, den)
        lat = latencies_by_label.get(label, [])
        summary[label] = {
            "n_matches": len(by_match), "n_controls": int(den.sum()),
            "detection_rate": round(est.value, 3) if np.isfinite(est.value) else None,
            "ci90": [round(est.lo, 3), round(est.hi, 3)] if np.isfinite(est.lo) else None,
            "mean_latency": round(float(np.mean(lat)), 2) if lat else None,
        }
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*", help="competition-season keys, e.g. 2_27")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lake = Lake.open()
    seasons = eligible_seasons(lake)
    if args.keys:
        seasons = seasons.filter(pl.col("key").is_in(args.keys))
    print(f"{seasons.height} eligible competition-seasons")

    t0 = time.time()
    all_team_matches = []
    for row in seasons.iter_rows(named=True):
        print(f"{row['competition']} {row['season']} ({row['key']})")
        all_team_matches += collect_team_matches(lake, row["competition_id"], row["season_id"], rng)
    print(f"\n{len(all_team_matches)} team-matches total, collected in {time.time() - t0:.0f}s")

    # Pass 1: one global threshold from every pooled null trajectory.
    all_nulls = [ns for tm in all_team_matches for ns in tm["null_scores"]]
    threshold = calibrate_threshold(all_nulls, target=FALSE_ALARM_TARGET,
                                     burn_in=BURN_IN, refractory=REFRACTORY)
    overall_null_rate = false_alarm_rate(all_nulls, threshold, BURN_IN, REFRACTORY)
    print(f"\ncalibrated threshold={threshold:.3f} "
          f"(target <= {FALSE_ALARM_TARGET} false alarms/team-match, got {overall_null_rate:.3f})")

    by_season_nulls: dict = {}
    for tm in all_team_matches:
        by_season_nulls.setdefault((tm["competition_id"], tm["season_id"]), []).extend(tm["null_scores"])
    for key, nulls in by_season_nulls.items():
        rate = false_alarm_rate(nulls, threshold, BURN_IN, REFRACTORY)
        print(f"  season {key}: null rate at calibrated threshold = {rate:.3f}")

    # Pass 2: alarms once per team-match, then check every control against them.
    rows_by_label: dict = {}
    latencies_by_label: dict = {}
    sub_split = {"with_sub_within_60s": [], "without_sub_within_60s": []}

    for tm in all_team_matches:
        alarm_idx = alarms(tm["real_scores"], threshold, BURN_IN, REFRACTORY)
        match_key = (tm["competition_id"], tm["season_id"], tm["match_id"], tm["team_id"])
        for c in tm["controls"]:
            result = check_control(tm["real_scores"], c["possession"], alarm_idx, WITHIN_POSSESSIONS)
            by_match = rows_by_label.setdefault(c["label"], {})
            d, n = by_match.get(match_key, (0, 0))
            by_match[match_key] = (d + int(result.detected), n + 1)
            if result.detected:
                latencies_by_label.setdefault(c["label"], []).append(result.latency)
            if c["label"] == "real_formation_change":
                bucket = "with_sub_within_60s" if c.get("near_sub") else "without_sub_within_60s"
                sub_split[bucket].append(int(result.detected))

    summary = summarise(rows_by_label, latencies_by_label)
    print("\n== detection rate by control type (90% CI, bootstrap by team-match) ==")
    for label, s in summary.items():
        print(f"  {label}: {json.dumps(s)}")

    print("\n== real formation changes, split by whether a sub happened within 60s ==")
    for bucket, vals in sub_split.items():
        rate = round(float(np.mean(vals)), 3) if vals else None
        print(f"  {bucket}: n={len(vals)}, detection_rate={rate}")

    results = {
        "params": {"hazard": HAZARD, "prior_strength": PRIOR_STRENGTH,
                   "change_point_window": CHANGE_POINT_WINDOW, "burn_in": BURN_IN,
                   "refractory": REFRACTORY, "within_possessions": WITHIN_POSSESSIONS,
                   "n_permutations": N_PERMUTATIONS, "false_alarm_target": FALSE_ALARM_TARGET,
                   "calibrated_threshold": round(threshold, 4),
                   "overall_null_rate": round(overall_null_rate, 4)},
        "by_season_null_rate": {f"{k[0]}_{k[1]}": round(v, 4)
                                 for k, v in ((key, false_alarm_rate(n, threshold, BURN_IN, REFRACTORY))
                                              for key, n in by_season_nulls.items())},
        "detection": summary,
        "real_formation_change_by_sub": {
            k: {"n": len(v), "detection_rate": round(float(np.mean(v)), 3) if v else None}
            for k, v in sub_split.items()
        },
    }
    Path("validation_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s, wrote validation_results.json")
