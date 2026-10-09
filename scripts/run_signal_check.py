"""Model-free signal check across the lake (see dynamic_shift.signal_check).

For each team-match, take every real formation change, same-formation
shift, red card, and several random moments; compare the 15 possessions
before vs after on coarse measures. Report, per class, the mean change
(90% bootstrap CI by team-match) and the AUC against the random baseline
(0.5 = indistinguishable from an arbitrary moment).

    python scripts/run_signal_check.py
    python scripts/run_signal_check.py --keys 37_4
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

from pitchgraph.chains.counts import CountStore  # noqa: E402
from pitchgraph.chains.grid import TACTICAL  # noqa: E402
from pitchgraph.chains.validate import prepare as prepare_league  # noqa: E402
from pitchgraph.data.lake import Lake  # noqa: E402
from pitchgraph.library.build import eligible_seasons  # noqa: E402

from dynamic_shift.ground_truth import red_cards, tactical_shifts  # noqa: E402
from dynamic_shift.observe import match_possession_counts  # noqa: E402
from dynamic_shift.signal_check import STAT_NAMES, auc_vs_baseline, boot_ci, event_index, window_stat  # noqa: E402

WINDOW = 15
RANDOM_POINTS_PER_MATCH = 5
LABELS = ("real_formation_change", "same_formation", "red_card", "random")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lake = Lake.open()
    seasons = eligible_seasons(lake)
    if args.keys:
        seasons = seasons.filter(pl.col("key").is_in(args.keys))
    print(f"{seasons.height} eligible competition-seasons")

    t0 = time.time()
    # label -> stat -> list of per-team-match arrays; plus flat values for AUC
    per_match = {lab: {s: [] for s in STAT_NAMES} for lab in LABELS}
    flat = {lab: {s: [] for s in STAT_NAMES} for lab in LABELS}
    n_events = {lab: 0 for lab in LABELS}

    for row in seasons.iter_rows(named=True):
        cid, sid = row["competition_id"], row["season_id"]
        events_df = (lake.events()
                     .filter((pl.col("competition_id") == cid) & (pl.col("season_id") == sid))
                     .collect())
        store = CountStore.from_events(events_df.lazy(), TACTICAL)
        _, teams, _, _ = prepare_league(store)
        by_match = {k[0]: df.lazy() for k, df in events_df.partition_by("match_id", as_dict=True).items()}
        print(f"{row['competition']} {row['season']} ({row['key']}): {len(teams)} teams "
              f"[{time.time() - t0:.0f}s]", flush=True)

        for team_id in teams:
            for match_id in store.team(team_id).match_ids.tolist():
                ev = by_match.get(match_id)
                if ev is None:
                    continue
                poss = match_possession_counts(ev, team_id, TACTICAL)
                if len(poss) < 2 * WINDOW:
                    continue
                ids = [p[0] for p in poss]

                events = {lab: [] for lab in LABELS}
                for s in tactical_shifts(ev, team_id):
                    events["real_formation_change" if s["real_change"] else "same_formation"].append(s["possession"])
                events["red_card"] = [c["possession"] for c in red_cards(ev) if c["team_id"] == team_id]
                span = np.arange(WINDOW, len(poss) - WINDOW + 1)
                if len(span):
                    k = min(RANDOM_POINTS_PER_MATCH, len(span))
                    events["random"] = [ids[i] for i in rng.choice(span, size=k, replace=False)]

                for lab, plist in events.items():
                    vals = {s: [] for s in STAT_NAMES}
                    for p in plist:
                        i = event_index(ids, p)
                        st = window_stat(poss, i, WINDOW, TACTICAL) if i is not None else None
                        if st is None:
                            continue
                        for s in STAT_NAMES:
                            vals[s].append(st[s])
                    if not vals[STAT_NAMES[0]]:
                        continue
                    n_events[lab] += len(vals[STAT_NAMES[0]])
                    for s in STAT_NAMES:
                        per_match[lab][s].append(np.array(vals[s]))
                        flat[lab][s].extend(vals[s])

    results = {"window": WINDOW, "random_points_per_match": RANDOM_POINTS_PER_MATCH, "classes": {}}
    print("\n== change in the 15 possessions after vs before, by class ==")
    for stat in STAT_NAMES:
        print(f"\n[{stat}]" + ("   <- pre-specified primary statistic" if stat == "tv_dest" else ""))
        base = np.array(flat["random"][stat])
        for lab in LABELS:
            mean, lo, hi = boot_ci(per_match[lab][stat], rng)
            auc = auc_vs_baseline(np.array(flat[lab][stat]), base) if lab != "random" else None
            print(f"  {lab:22s} n_events={n_events[lab]:5d} mean={mean:.4f} (90% CI {lo:.4f}-{hi:.4f})"
                  + (f"  AUC vs random={auc:.3f}" if auc is not None else ""))
            results["classes"].setdefault(lab, {})[stat] = {
                "n_events": n_events[lab], "mean": round(mean, 5),
                "ci90": [round(lo, 5), round(hi, 5)],
                "auc_vs_random": None if auc is None else round(auc, 4),
            }

    Path("signal_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s, wrote signal_check_results.json")
