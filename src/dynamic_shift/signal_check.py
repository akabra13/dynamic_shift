"""Model-free check: do real tactical changes shift coarse in-possession
behaviour more than arbitrary moments do?

No change-point model here on purpose. For an event at possession index i
we compare the `window` possessions before it with the `window` after, on
a coarse vocabulary (destination zone binned to thirds x lanes = 15 cells,
plus shot / turnover / actions-per-possession rates). The same statistic
is computed at random moments of the same matches, so "does a real change
stand out" has a baseline.

Pre-specified primary statistic: total-variation distance between the
before/after 15-cell destination distributions. The rest are secondary.
"""
from __future__ import annotations

import numpy as np

from pitchgraph.chains.grid import TACTICAL, Grid

from dynamic_shift.describe import zone_lane
from pitchgraph.graphs.routes import LANES

THIRD_EDGES = (40.0, 80.0)
STAT_NAMES = ("tv_dest", "d_shot_rate", "d_lose_rate", "d_actions")


def coarse_cell_of_zone(grid: Grid = TACTICAL) -> np.ndarray:
    """Each zone's index into the 15-cell (third x lane) vocabulary."""
    lanes = zone_lane(grid)
    cells = []
    for (x, _), lane in zip(grid.centres(), lanes):
        third = int(x >= THIRD_EDGES[0]) + int(x >= THIRD_EDGES[1])
        cells.append(third * len(LANES) + LANES.index(lane))
    return np.array(cells, dtype=int)


def window_features(counts: list[np.ndarray], grid: Grid = TACTICAL) -> dict:
    """Aggregate a window of possession count vectors into coarse features."""
    n = grid.n
    C = np.sum(counts, axis=0).reshape(n, n + 2)
    moves = C[:, :n].sum(axis=0)  # moves landing in each destination zone
    cell = coarse_cell_of_zone(grid)
    dest = np.bincount(cell, weights=moves, minlength=3 * len(LANES))
    total_moves = dest.sum()
    n_poss = max(len(counts), 1)
    return {
        "dest": dest / total_moves if total_moves > 0 else None,
        "shot_rate": float(C[:, n].sum()) / n_poss,
        "lose_rate": float(C[:, n + 1].sum()) / n_poss,
        "actions": float(C.sum()) / n_poss,
    }


def compare_windows(before: list[np.ndarray], after: list[np.ndarray],
                    grid: Grid = TACTICAL) -> dict | None:
    """Before/after change statistics (see STAT_NAMES), or None if either
    window has no completed moves to form a destination distribution."""
    fb, fa = window_features(before, grid), window_features(after, grid)
    if fb["dest"] is None or fa["dest"] is None:
        return None
    return {
        "tv_dest": 0.5 * float(np.abs(fa["dest"] - fb["dest"]).sum()),
        "d_shot_rate": abs(fa["shot_rate"] - fb["shot_rate"]),
        "d_lose_rate": abs(fa["lose_rate"] - fb["lose_rate"]),
        "d_actions": abs(fa["actions"] - fb["actions"]),
    }


def event_index(possession_ids: list[int], event_possession: int) -> int | None:
    """Index of the team's first possession with id >= the event's."""
    for i, p in enumerate(possession_ids):
        if p >= event_possession:
            return i
    return None


def window_stat(possessions: list[tuple[int, int, np.ndarray]], i: int, window: int,
                grid: Grid = TACTICAL) -> dict | None:
    """Statistics for an event at possession index `i`: the `window`
    possessions before vs from `i` onward. Both windows must sit inside
    one half (so a half-time break never counts as a 'change') and be
    complete."""
    if i < window or i + window > len(possessions):
        return None
    before, after = possessions[i - window:i], possessions[i:i + window]
    if len({p[1] for p in before}) != 1 or len({p[1] for p in after}) != 1:
        return None
    if before[0][1] != after[0][1]:
        return None
    return compare_windows([p[2] for p in before], [p[2] for p in after], grid)


def auc_vs_baseline(values: np.ndarray, baseline: np.ndarray) -> float:
    """P(value > baseline draw), ties counted half. 0.5 = indistinguishable."""
    values, baseline = np.asarray(values, float), np.asarray(baseline, float)
    if len(values) == 0 or len(baseline) == 0:
        return float("nan")
    gt = (values[:, None] > baseline[None, :]).mean()
    eq = (values[:, None] == baseline[None, :]).mean()
    return float(gt + 0.5 * eq)


def boot_ci(per_match: list[np.ndarray], rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    """Mean of all event values, 90% CI resampling whole team-matches."""
    pooled = np.concatenate(per_match) if per_match else np.array([])
    if len(pooled) == 0:
        return float("nan"), float("nan"), float("nan")
    sums = np.array([a.sum() for a in per_match])
    cnts = np.array([len(a) for a in per_match], dtype=float)
    ms = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per_match), len(per_match))
        ms.append(sums[idx].sum() / max(cnts[idx].sum(), 1))
    return float(pooled.mean()), float(np.percentile(ms, 5)), float(np.percentile(ms, 95))
