"""Pre-registered test: persistent-homology summaries of 360 freeze-frame
team shape around labelled tactical changes vs random moments.

    python scripts/run_topology_check.py [--keys 43_106]
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
from dynamic_shift.report import family_report  # noqa: E402
from dynamic_shift.shape import TOPO_NAMES, random_times, shape_change, team_topology_rows  # noqa: E402

WINDOW_S = 300.0
RANDOM_POINTS_PER_MATCH = 5
LABELS = ("real_substantive", "real_clean", "same_formation", "random")


def event_time(me: pl.DataFrame, team_id: int, possession: int):
    r = (me.filter((pl.col("type") == "Tactical Shift") & (pl.col("team_id") == team_id)
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
    m = lake.matches().filter(pl.col("has_360")).with_columns(
        key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    if args.keys:
        m = m.filter(pl.col("key").is_in(args.keys))
    seasons = m.group_by("key", "competition", "season").agg(pl.len().alias("n")).sort("key")

    t0 = time.time()
    topo = {lab: [] for lab in LABELS}
    key_ids: dict = {}
    for r in seasons.iter_rows(named=True):
        ids = m.filter(pl.col("key") == r["key"])["match_id"].to_list()
        ev = lake.scan("events").filter(pl.col("match_id").is_in(ids)).collect()
        fr = lake.scan("frames").filter(pl.col("match_id").is_in(ids)).collect()
        ev_by = {k[0]: d for k, d in ev.partition_by("match_id", as_dict=True).items()}
        fr_by = {k[0]: d for k, d in fr.partition_by("match_id", as_dict=True).items()}
        print(f"{r['competition']} {r['season']}: {len(ids)} matches [{time.time() - t0:.0f}s]", flush=True)
        for mid in ids:
            me, mf = ev_by.get(mid), fr_by.get(mid)
            if me is None or mf is None:
                continue
            lazy = me.lazy()
            for team_id in me["team_id"].drop_nulls().unique().to_list():
                rows = team_topology_rows(mf, me, team_id)
                if rows.height < 100:
                    continue
                key = key_ids.setdefault((r["key"], mid, team_id), len(key_ids))
                points = {lab: [] for lab in LABELS}
                for lab, lst in classify_shifts(shift_details(lazy, team_id)).items():
                    points[lab] = [pt for d in lst if (pt := event_time(me, team_id, d["possession"]))]
                points["random"] = random_times(rows, RANDOM_POINTS_PER_MATCH, rng, WINDOW_S)
                for lab, plist in points.items():
                    for period, t in plist:
                        ch = shape_change(rows, period, t, WINDOW_S, features=TOPO_NAMES, signed=True)
                        if ch:
                            topo[lab].append((key, ch))

    print("\nevents with usable windows: " + str({l: len(topo[l]) for l in LABELS}))
    feats = tuple(f"{ph}_{f}" for ph in ("oop", "ip") for f in TOPO_NAMES)
    results = {"window_seconds": WINDOW_S,
               "topology": family_report("freeze-frame topology", topo, feats, rng)}
    Path("topology_check_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
