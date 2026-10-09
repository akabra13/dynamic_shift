"""Before/after possession-count comparison, on hand-traceable data."""
import numpy as np
import pytest

from pitchgraph.chains.grid import TACTICAL, Grid

from dynamic_shift.describe import (
    describe_change, lane_shares, shots_per_possession, top_edge_shifts, zone_lane,
)


def _counts(grid, moves: dict, shots: float = 0.0, loses: float = 0.0) -> np.ndarray:
    """Build a flattened (z0, out) count vector by hand. Shots/loses are all
    attributed to zone 0, which is all these tests need."""
    n = grid.n
    C = np.zeros((n, n + 2))
    for (z0, z1), c in moves.items():
        C[z0, z1] = c
    C[0, n] = shots
    C[0, n + 1] = loses
    return C.reshape(-1)


def _zone_in(grid, x: float, y: float) -> int:
    return int(grid.index(np.array([x]), np.array([y]))[0])


def test_zone_lane_uses_grid_centre_y():
    # TACTICAL's y-edges are the lane edges, so it's lane-aligned by design.
    lanes = zone_lane(TACTICAL)
    assert len(lanes) == TACTICAL.n
    wide_left_zone = _zone_in(TACTICAL, x=10.0, y=5.0)
    wide_right_zone = _zone_in(TACTICAL, x=10.0, y=75.0)
    assert lanes[wide_left_zone] == "wide_left"
    assert lanes[wide_right_zone] == "wide_right"


def test_lane_shares_from_move_destinations():
    z_left, z_right = _zone_in(TACTICAL, 10.0, 5.0), _zone_in(TACTICAL, 10.0, 75.0)
    counts = _counts(TACTICAL, moves={(z_left, z_right): 3, (z_right, z_right): 1})
    shares = lane_shares(counts, TACTICAL)
    assert shares["wide_right"] == pytest.approx(1.0)
    assert shares["wide_left"] == pytest.approx(0.0)


def test_shots_per_possession():
    grid = Grid.uniform(2, 1)
    counts = _counts(grid, moves={}, shots=4.0)
    assert shots_per_possession(counts, grid, n_possessions=2) == pytest.approx(2.0)


def test_top_edge_shifts_finds_the_biggest_absolute_change():
    # Three destinations, not two, so the shift isn't a symmetric tie: zone
    # 2's share grows the most (+0.5) and must win over zone 0 (-0.3) or 1 (-0.2).
    grid = Grid.uniform(3, 1)
    before = _counts(grid, moves={(0, 0): 5, (0, 1): 4, (0, 2): 1})
    after = _counts(grid, moves={(0, 0): 2, (0, 1): 2, (0, 2): 6})
    top = top_edge_shifts(before, after, grid, k=1)[0]
    assert (top["from_zone"], top["to_zone"]) == (0, 2)
    assert top["before"] == pytest.approx(0.1)
    assert top["after"] == pytest.approx(0.6)


def test_describe_change_reports_lane_and_shot_rate_shifts():
    # Three lanes, not two, for the same reason: a plain two-lane swap ties
    # in absolute shift and the choice of which lane to report is arbitrary.
    z_src = _zone_in(TACTICAL, 10.0, 40.0)
    z_left = _zone_in(TACTICAL, 10.0, 5.0)
    z_centre = _zone_in(TACTICAL, 10.0, 40.0)
    z_right = _zone_in(TACTICAL, 10.0, 75.0)
    before = _counts(TACTICAL, moves={(z_src, z_left): 5, (z_src, z_centre): 4,
                                       (z_src, z_right): 1}, shots=1.0)
    after = _counts(TACTICAL, moves={(z_src, z_left): 1, (z_src, z_centre): 2,
                                      (z_src, z_right): 7}, shots=3.0)
    sentences = describe_change(before, after, TACTICAL, minute=63.0)
    assert any("wide right" in s for s in sentences)
    assert any("more shots per possession" in s for s in sentences)
    assert all(s.startswith("From 63") for s in sentences)


def test_describe_change_is_empty_below_thresholds():
    grid = Grid.uniform(2, 1)
    before = _counts(grid, moves={(0, 0): 100, (0, 1): 100})
    after = _counts(grid, moves={(0, 0): 101, (0, 1): 100})  # negligible shift
    assert describe_change(before, after, grid) == []
