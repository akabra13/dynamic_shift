"""Pre-registered test: pass geometry and passing-network features around
labelled tactical changes vs random moments. See README ("Pre-registered").

    python scripts/run_graph_check.py [--keys 37_4]
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

from dynamic_shift.geometry import GEOMETRY_NAMES, NETWORK_NAMES, network_features, pass_geometry  # noqa: E402
from dynamic_shift.ground_truth import classify_shifts, shift_details  # noqa: E402
from dynamic_shift.observe import match_possession_counts  # noqa: E402
from dynamic_shift.report import family_report  # noqa: E402
from dynamic_shift.signal_check import event_index, window_ids  # noqa: E402

WINDOW = 15
RANDOM_POINTS_PER_MATCH = 5
LABELS = ("real_substantive", "real_clean", "same_formation", "random")


def window_feats(passes: pl.DataFrame, ids: list[int]):
    w = passes.filter(pl.col("possession").is_in(ids))
    arr = w.filter(pl.col("end_x").is_not_null()).select("x", "y", "end_x", "end_y").to_numpy()
    done = w.filter(pl.col("outcome").is_null() & pl.col("recipient_id").is_not_null())
    return (pass_geometry(arr),
            network_features(done["player_id"].to_numpy(), done["recipient_id"].to_numpy()))


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
    geo = {lab: [] for lab in LABELS}
    net = {lab: [] for lab in LABELS}
    key_ids: dict = {}

    for row in seasons.iter_rows(named=True):
        cid, sid = row["competition_id"], row["season_id"]
        ev_df = (lake.events()
                 .filter((pl.col("competition_id") == cid) & (pl.col("season_id") == sid))
                 .collect())
        store = CountStore.from_events(ev_df.lazy(), TACTICAL)
        _, teams, _, _ = prepare_league(store)
        by_match = {k[0]: df for k, df in ev_df.partition_by("match_id", as_dict=True).items()}
        print(f"{row['competition']} {row['season']} ({row['key']}): {len(teams)} teams "
              f"[{time.time() - t0:.0f}s]", flush=True)

        for team_id in teams:
            for match_id in store.team(team_id).match_ids.tolist():
                me = by_match.get(match_id)
                if me is None:
                    continue
                lazy = me.lazy()
                poss = match_possession_counts(lazy, team_id, TACTICAL)
                if len(poss) < 2 * WINDOW:
                    continue
                ids = [p[0] for p in poss]
                passes = me.filter((pl.col("type") == "Pass") & (pl.col("team_id") == team_id))
                key = key_ids.setdefault((row["key"], match_id, team_id), len(key_ids))

                points = {lab: [] for lab in LABELS}
                for lab, lst in classify_shifts(shift_details(lazy, team_id)).items():
                    points[lab] = [d["possession"] for d in lst]
                span = np.arange(WINDOW, len(poss) - WINDOW + 1)
                if len(span):
                    k = min(RANDOM_POINTS_PER_MATCH, len(span))
                    points["random"] = [ids[i] for i in rng.choice(span, size=k, replace=False)]

                for lab, plist in points.items():
                    for p in plist:
                        i = event_index(ids, p)
                        w = window_ids(poss, i, WINDOW) if i is not None else None
                        if w is None:
                            continue
                        gb, nb = window_feats(passes, w[0])
                        ga, na = window_feats(passes, w[1])
                        if gb and ga:
                            geo[lab].append((key, {f: ga[f] - gb[f] for f in GEOMETRY_NAMES}))
                        if nb and na:
                            net[lab].append((key, {f: na[f] - nb[f] for f in NETWORK_NAMES}))

    print("\nevents with usable windows: geometry " + str({l: len(geo[l]) for l in LABELS})
          + "  network " + str({l: len(net[l]) for l in LABELS}))
    results = {
        "window_possessions": WINDOW,
        "geometry": family_report("pass geometry", geo, GEOMETRY_NAMES, rng),
        "network": family_report("passing network", net, NETWORK_NAMES, rng),
    }
    Path("graph_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
