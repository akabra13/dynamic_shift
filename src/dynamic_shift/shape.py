"""Team shape from StatsBomb 360 freeze-frames, and a before/after check.

A freeze-frame lists the players the broadcast camera can see at an
event. For a team T we keep T's visible outfield players, mirrored so T
always attacks toward x=120, and summarise each frame:

  depth      mean x of T's visible outfield players (block height)
  depth_rel  depth minus the ball's x (removes where the ball is, which
             drives who the camera sees)
  width      std of y
  compact    mean distance to the players' centroid

Frames are split by phase: out of possession (`oop`, T's shape when
defending) and in possession. Visible-only players are a biased sample of
the team, so these are noisy summaries of shape, not exact line heights;
the before/after comparison uses the same bias on both sides.

Pre-specified primary statistic: |change| in `oop_depth_rel`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

PITCH_X, PITCH_Y = 120.0, 80.0
FEATURES = ("depth", "depth_rel", "width", "compact")
STAT_NAMES = tuple(f"{phase}_{f}" for phase in ("oop", "ip") for f in FEATURES)
MIN_OUTFIELD_VISIBLE = 6
MIN_FRAMES_PER_WINDOW = 20


def team_shape_rows(frames: pl.DataFrame, events: pl.DataFrame, team_id: int) -> pl.DataFrame:
    """One row per usable freeze-frame for `team_id`.

    frames: match_id, event_id, teammate, keeper, x, y (one row per player)
    events: match_id, event_id, team_id, possession_team_id, period, period_seconds, x, y
    Returns: period, t, oop, n_vis, depth, depth_rel, width, compact
    """
    ev = events.select(
        "match_id", "event_id", "period", "period_seconds", "possession_team_id",
        pl.col("team_id").alias("ev_team"), pl.col("x").alias("bx"), pl.col("y").alias("by"),
    ).filter(pl.col("bx").is_not_null())
    df = frames.join(ev, on=["match_id", "event_id"], how="inner")
    df = df.filter(
        (pl.col("teammate") == (pl.col("ev_team") == team_id)) & (~pl.col("keeper"))
    )
    flip = pl.col("ev_team") != team_id
    df = df.with_columns(
        px=pl.when(flip).then(PITCH_X - pl.col("x")).otherwise(pl.col("x")),
        py=pl.when(flip).then(PITCH_Y - pl.col("y")).otherwise(pl.col("y")),
        bx=pl.when(flip).then(PITCH_X - pl.col("bx")).otherwise(pl.col("bx")),
    )
    df = df.with_columns(
        cx=pl.col("px").mean().over("event_id"), cy=pl.col("py").mean().over("event_id"),
    ).with_columns(
        d=((pl.col("px") - pl.col("cx")) ** 2 + (pl.col("py") - pl.col("cy")) ** 2).sqrt(),
    )
    out = (
        df.group_by("event_id")
          .agg(
              pl.first("period"), pl.first("period_seconds").alias("t"),
              (pl.first("possession_team_id") != team_id).alias("oop"),
              pl.len().alias("n_vis"),
              pl.col("px").mean().alias("depth"),
              (pl.col("px") - pl.col("bx")).mean().alias("depth_rel"),
              pl.col("py").std().alias("width"),
              pl.col("d").mean().alias("compact"),
          )
          .filter(pl.col("n_vis") >= MIN_OUTFIELD_VISIBLE)
          .drop_nulls()
          .sort(["period", "t"])
    )
    return out


def _window_mean(rows: pl.DataFrame, period: int, t0: float, t1: float, oop: bool) -> dict | None:
    w = rows.filter((pl.col("period") == period) & (pl.col("t") >= t0) & (pl.col("t") < t1)
                    & (pl.col("oop") == oop))
    if w.height < MIN_FRAMES_PER_WINDOW:
        return None
    return {f: float(w[f].mean()) for f in FEATURES}


def shape_change(rows: pl.DataFrame, period: int, t: float, window_s: float = 300.0) -> dict:
    """|mean feature after - before| for each (phase, feature), over the
    `window_s` seconds either side of `t` inside one period. Entries are
    present only where both sides have enough frames."""
    out = {}
    for oop, phase in ((True, "oop"), (False, "ip")):
        before = _window_mean(rows, period, t - window_s, t, oop)
        after = _window_mean(rows, period, t, t + window_s, oop)
        if before is None or after is None:
            continue
        for f in FEATURES:
            out[f"{phase}_{f}"] = abs(after[f] - before[f])
    return out


def random_times(rows: pl.DataFrame, k: int, rng: np.random.Generator,
                 window_s: float = 300.0) -> list[tuple[int, float]]:
    """k random (period, t) points that have room for a full window either side."""
    cands = []
    for period in rows["period"].unique().to_list():
        ts = rows.filter(pl.col("period") == period)["t"]
        lo, hi = float(ts.min()) + window_s, float(ts.max()) - window_s
        if hi > lo:
            cands.append((period, lo, hi))
    if not cands:
        return []
    picks = []
    for _ in range(k):
        period, lo, hi = cands[rng.integers(len(cands))]
        picks.append((int(period), float(rng.uniform(lo, hi))))
    return picks


TOPO_NAMES = ("h0_sum", "h0_max", "h1_sum", "h1_n")


def topology_features(points: np.ndarray) -> dict | None:
    """Persistent-homology summary of a frame's point cloud (Vietoris-Rips).

    H0 deaths are the minimum-spanning-tree edge lengths: h0_sum is the
    total, h0_max the single biggest gap (how detached the most isolated
    player is). H1 are loops -- rings of players around an empty space:
    h1_sum the total persistence, h1_n how many.
    """
    from ripser import ripser
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    pts = np.asarray(points, dtype=float)
    if len(pts) < MIN_OUTFIELD_VISIBLE:
        return None
    D = squareform(pdist(pts))
    mst = minimum_spanning_tree(D).data
    dgm1 = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    pers = dgm1[:, 1] - dgm1[:, 0] if len(dgm1) else np.array([])
    return {"h0_sum": float(mst.sum()), "h0_max": float(mst.max()),
            "h1_sum": float(pers.sum()), "h1_n": float(len(pers))}
