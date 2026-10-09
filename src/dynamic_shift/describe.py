"""Before/after possession-count comparison -> plain-English change sentences.

Takes two flattened (start zone, outcome) count vectors in the same
category order as `observe.categories()` -- e.g. two windows of a team's
possessions either side of a detected change point -- and describes the
biggest shifts: which lane the ball entered through, the shot rate per
possession, and which zone-to-zone move changed the most.

Scope note: the BOCPD observation model (monitor.py) only tracks
categorical counts, not each shot's xG, so a true expected-goals
"threat per possession" (pitchgraph.chains.team.Chain.threat) isn't
recoverable from these vectors alone -- shots per possession stands in
for it here. A real xG-weighted comparison needs the actual events for
the before/after windows, which is export.py's job once attribution is
wired up.
"""
from __future__ import annotations

import numpy as np

from pitchgraph.chains.grid import TACTICAL, Grid
from pitchgraph.graphs.routes import LANE_EDGES, LANES

LANE_SHIFT_THRESHOLD = 0.05
EDGE_SHIFT_THRESHOLD = 0.02


def _lane_of(y: float) -> str:
    e = LANE_EDGES
    if y < e[0]:
        return LANES[0]
    if y < e[1]:
        return LANES[1]
    if y < e[2]:
        return LANES[2]
    if y < e[3]:
        return LANES[3]
    return LANES[4]


def zone_lane(grid: Grid = TACTICAL) -> list[str]:
    """Each zone's lane, by its centre y-coordinate."""
    return [_lane_of(y) for _, y in grid.centres()]


def _unflatten(counts: np.ndarray, grid: Grid) -> np.ndarray:
    n = grid.n
    return np.asarray(counts, dtype=float).reshape(n, n + 2)


def lane_shares(counts: np.ndarray, grid: Grid = TACTICAL) -> dict[str, float]:
    """Share of completed moves landing in each lane, by destination zone."""
    C = _unflatten(counts, grid)
    n = grid.n
    moves = C[:, :n]
    total = moves.sum()
    shares = {lane: 0.0 for lane in LANES}
    if total == 0:
        return shares
    lanes = zone_lane(grid)
    dest_totals = moves.sum(axis=0)
    for z1, lane in enumerate(lanes):
        shares[lane] += dest_totals[z1] / total
    return shares


def shots_per_possession(counts: np.ndarray, grid: Grid = TACTICAL,
                          n_possessions: int = 1) -> float:
    C = _unflatten(counts, grid)
    n = grid.n
    return float(C[:, n].sum()) / max(n_possessions, 1)


def top_edge_shifts(before: np.ndarray, after: np.ndarray, grid: Grid = TACTICAL,
                     k: int = 3) -> list[dict]:
    """The k zone-to-zone moves whose share of all moves changed the most."""
    Cb, Ca = _unflatten(before, grid), _unflatten(after, grid)
    n = grid.n
    mb, ma = Cb[:, :n], Ca[:, :n]
    tb, ta = mb.sum(), ma.sum()
    pb = mb / tb if tb > 0 else np.zeros_like(mb)
    pa = ma / ta if ta > 0 else np.zeros_like(ma)
    delta = pa - pb
    flat_idx = np.argsort(-np.abs(delta), axis=None)[:k]
    out = []
    for idx in flat_idx:
        z0, z1 = divmod(int(idx), n)
        out.append({"from_zone": z0, "to_zone": z1,
                     "before": float(pb[z0, z1]), "after": float(pa[z0, z1])})
    return out


def describe_change(before: np.ndarray, after: np.ndarray, grid: Grid = TACTICAL,
                     minute: float | None = None) -> list[str]:
    """Plain-English sentences describing the biggest shifts between two
    possession-count windows, most notable first. Empty if nothing crosses
    the reporting thresholds."""
    sentences = []
    prefix = f"From {minute:.0f}', " if minute is not None else ""

    lanes_before, lanes_after = lane_shares(before, grid), lane_shares(after, grid)
    biggest_lane = max(LANES, key=lambda lane: abs(lanes_after[lane] - lanes_before[lane]))
    lane_delta = lanes_after[biggest_lane] - lanes_before[biggest_lane]
    if abs(lane_delta) >= LANE_SHIFT_THRESHOLD:
        direction = "up" if lane_delta > 0 else "down"
        sentences.append(
            f"{prefix}their build-up moved toward the {biggest_lane.replace('_', ' ')}: "
            f"{lanes_after[biggest_lane]:.0%} of entries there, "
            f"{direction} from {lanes_before[biggest_lane]:.0%}."
        )

    spp_before = shots_per_possession(before, grid)
    spp_after = shots_per_possession(after, grid)
    if abs(spp_after - spp_before) > 1e-9:
        direction = "more" if spp_after > spp_before else "fewer"
        sentences.append(
            f"{prefix}{direction} shots per possession: {spp_after:.2f}, vs {spp_before:.2f} before."
        )

    top_edge = top_edge_shifts(before, after, grid, k=1)[0]
    if abs(top_edge["after"] - top_edge["before"]) >= EDGE_SHIFT_THRESHOLD:
        direction = "more" if top_edge["after"] > top_edge["before"] else "less"
        sentences.append(
            f"{prefix}zone {top_edge['from_zone']} -> {top_edge['to_zone']} is used {direction} often: "
            f"{top_edge['after']:.0%} of their moves from there, vs {top_edge['before']:.0%} before."
        )

    return sentences
