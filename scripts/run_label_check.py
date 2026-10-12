"""How often is a StatsBomb 'real formation change' an actual change in who plays where?

For every Tactical Shift whose formation label differs from the previous
one, compare the lineup before and after (players on the pitch both
times, by position name). Reports the distribution of `moved` (continuing
players whose position changed) so the validation controls can be split
into substantive reshuffles vs relabels / pure substitutions.

    python scripts/run_label_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from dynamic_shift.ground_truth import shift_details  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    lake = Lake.open()
    t0 = time.time()
    ev = (lake.scan("events")
          .filter(pl.col("type").is_in(["Starting XI", "Tactical Shift"]))
          .collect())
    print(f"{ev.height} tactics events [{time.time() - t0:.0f}s]")

    rows = []
    for (mid, tid), g in ev.partition_by("match_id", "team_id", as_dict=True).items():
        for d in shift_details(g.lazy(), tid):
            rows.append({**d, "match_id": mid})
    df = pl.DataFrame(rows)
    print(f"{df.height} tactical shifts in total")

    same = df.filter(~pl.col("real_change"))
    real = df.filter(pl.col("real_change"))
    n = real.height
    print(f"\nlabelled 'real' formation changes: {n}   same-formation shifts: {same.height}")

    def share(mask):
        return round(float(real.filter(mask).height) / n, 3)

    bins = {
        "moved == 0 (relabel or pure sub)": share(pl.col("moved") == 0),
        "moved 1-2": share(pl.col("moved").is_between(1, 2)),
        "moved 3-5": share(pl.col("moved").is_between(3, 5)),
        "moved >= 6": share(pl.col("moved") >= 6),
        "someone joined/left (sub or card)": share((pl.col("joined") > 0) | (pl.col("left") > 0)),
        "moved >= 3 and nobody joined/left": share((pl.col("moved") >= 3) & (pl.col("joined") == 0)
                                                     & (pl.col("left") == 0)),
    }
    print("\nshare of labelled real changes:")
    for k, v in bins.items():
        print(f"  {k}: {v}")
    print(f"\nmoved: mean={real['moved'].mean():.2f} median={real['moved'].median()} "
          f"p90={np.percentile(real['moved'].to_numpy(), 90):.0f}")
    print(f"same-formation shifts, moved: mean={same['moved'].mean():.2f} "
          f"share moved==0: {round(float(same.filter(pl.col('moved') == 0).height) / max(same.height, 1), 3)}")

    top = (real.group_by(["formation_before", "formation_after"]).agg(pl.len().alias("n"),
                                                                        pl.col("moved").mean().alias("moved"))
               .sort("n", descending=True).head(12))
    print("\nmost common formation transitions (n, mean moved):")
    for r in top.iter_rows(named=True):
        print(f"  {r['formation_before']} -> {r['formation_after']}: n={r['n']}, moved={r['moved']:.1f}")

    Path("label_check_results.json").write_text(json.dumps({
        "n_real": n, "n_same": same.height, "share": bins,
        "moved_mean_real": float(real["moved"].mean()),
        "moved_mean_same": float(same["moved"].mean()),
    }, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
