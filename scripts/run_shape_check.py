"""Does a real tactical change move team SHAPE (360 freeze-frames) more than
a random moment does?

Only matches with 360 data are used. For each team-match, shape features
(see dynamic_shift.shape) are averaged over the 5 minutes before vs after
each real formation change, same-formation shift, red card, and several
random moments, inside one half. Per class: mean |change| with a 90%
bootstrap CI by team-match, and AUC against the random baseline
(0.5 = indistinguishable from an arbitrary moment).

    python scripts/run_shape_check.py
    python scripts/run_shape_check.py --keys 43_106
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

from dynamic_shift.ground_truth import red_cards, tactical_shifts  # noqa: E402
from dynamic_shift.shape import STAT_NAMES, random_times, shape_change, team_shape_rows  # noqa: E402
from dynamic_shift.signal_check import auc_vs_baseline, boot_ci  # noqa: E402

WINDOW_S = 300.0
RANDOM_POINTS_PER_MATCH = 5
LABELS = ("real_formation_change", "same_formation", "red_card", "random")
PRIMARY = "oop_depth_rel"


def event_time(match_events: pl.DataFrame, team_id: int, possession: int, typ: str):
    """(period, period_seconds) of the first matching event in that possession."""
    r = (match_events.filter((pl.col("type") == typ) & (pl.col("team_id") == team_id)
                             & (pl.col("possession") == possession))
                     .sort("idx").select("period", "period_seconds").head(1))
    return (int(r["period"][0]), float(r["period_seconds"][0])) if r.height else None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lake = Lake.open()
    if not lake.has("frames"):
        sys.exit("lake has no frames table")

    matches = lake.matches()
    m360 = matches.filter(pl.col("has_360"))
    m360 = m360.with_columns(key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    if args.keys:
        m360 = m360.filter(pl.col("key").is_in(args.keys))
    seasons = m360.group_by("key", "competition", "season").agg(pl.len().alias("n")).sort("key")
    print("360 seasons:")
    for r in seasons.iter_rows(named=True):
        print(f"  {r['key']}: {r['competition']} {r['season']} ({r['n']} matches)")

    t0 = time.time()
    per_match = {lab: {s: [] for s in STAT_NAMES} for lab in LABELS}
    flat = {lab: {s: [] for s in STAT_NAMES} for lab in LABELS}
    n_events = {lab: 0 for lab in LABELS}
    n_team_matches = 0

    for r in seasons.iter_rows(named=True):
        ids = m360.filter(pl.col("key") == r["key"])["match_id"].to_list()
        ev = (lake.scan("events").filter(pl.col("match_id").is_in(ids)).collect())
        fr = (lake.scan("frames").filter(pl.col("match_id").is_in(ids)).collect())
        print(f"{r['competition']} {r['season']}: {len(ids)} matches, {fr.height} player rows "
              f"[{time.time() - t0:.0f}s]", flush=True)
        ev_by = {k[0]: d for k, d in ev.partition_by("match_id", as_dict=True).items()}
        fr_by = {k[0]: d for k, d in fr.partition_by("match_id", as_dict=True).items()}

        for mid in ids:
            me, mf = ev_by.get(mid), fr_by.get(mid)
            if me is None or mf is None:
                continue
            lazy = me.lazy()
            for team_id in me["team_id"].drop_nulls().unique().to_list():
                rows = team_shape_rows(mf, me, team_id)
                if rows.height < 100:
                    continue
                n_team_matches += 1

                points = {lab: [] for lab in LABELS}
                for s in tactical_shifts(lazy, team_id):
                    pt = event_time(me, team_id, s["possession"], "Tactical Shift")
                    if pt:
                        points["real_formation_change" if s["real_change"] else "same_formation"].append(pt)
                for c in red_cards(lazy):
                    if c["team_id"] == team_id:
                        pt = event_time(me, team_id, c["possession"], "Foul Committed") \
                             or event_time(me, team_id, c["possession"], "Bad Behaviour")
                        if pt:
                            points["red_card"].append(pt)
                points["random"] = random_times(rows, RANDOM_POINTS_PER_MATCH, rng, WINDOW_S)

                for lab, plist in points.items():
                    vals = {s: [] for s in STAT_NAMES}
                    for period, t in plist:
                        for stat, v in shape_change(rows, period, t, WINDOW_S).items():
                            vals[stat].append(v)
                    for stat in STAT_NAMES:
                        if vals[stat]:
                            per_match[lab][stat].append(np.array(vals[stat]))
                            flat[lab][stat].extend(vals[stat])
                    if vals[PRIMARY]:
                        n_events[lab] += len(vals[PRIMARY])

    print(f"\n{n_team_matches} team-matches with enough 360 frames")
    results = {"window_seconds": WINDOW_S, "primary": PRIMARY, "n_team_matches": n_team_matches,
               "n_events_primary": n_events, "classes": {}}
    print(f"\n== |change| in team shape, 5 min after vs before ==  (events with usable windows, primary stat: {n_events})")
    for stat in STAT_NAMES:
        print(f"\n[{stat}]" + ("   <- pre-specified primary statistic" if stat == PRIMARY else ""))
        base = np.array(flat["random"][stat])
        for lab in LABELS:
            mean, lo, hi = boot_ci(per_match[lab][stat], rng)
            n = len(flat[lab][stat])
            auc = auc_vs_baseline(np.array(flat[lab][stat]), base) if lab != "random" else None
            print(f"  {lab:22s} n={n:5d} mean={mean:.3f} (90% CI {lo:.3f}-{hi:.3f})"
                  + (f"  AUC vs random={auc:.3f}" if auc is not None else ""))
            results["classes"].setdefault(lab, {})[stat] = {
                "n": n, "mean": None if np.isnan(mean) else round(mean, 4),
                "ci90": None if np.isnan(lo) else [round(lo, 4), round(hi, 4)],
                "auc_vs_random": None if auc is None or np.isnan(auc) else round(auc, 4),
            }

    Path("shape_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s, wrote shape_check_results.json")
