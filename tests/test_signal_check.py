"""Model-free before/after statistics, on hand-traceable data."""
import numpy as np

from pitchgraph.chains.grid import TACTICAL

from dynamic_shift.signal_check import (
    auc_vs_baseline, coarse_cell_of_zone, compare_windows, event_index, window_stat,
)

N = TACTICAL.n


def _zone_in(x, y):
    return int(TACTICAL.index(np.array([x]), np.array([y]))[0])


def _move(z0, z1, c=1.0, shots=0.0, loses=0.0):
    C = np.zeros((N, N + 2))
    C[z0, z1] = c
    C[0, N] = shots
    C[0, N + 1] = loses
    return C.reshape(-1)


def test_coarse_cells_cover_fifteen():
    cells = coarse_cell_of_zone()
    assert len(cells) == N
    assert set(cells) == set(range(15))


def test_identical_windows_have_zero_change():
    w = [_move(0, 7)] * 5
    stats = compare_windows(w, w)
    assert all(v == 0 for v in stats.values())


def test_moving_to_a_different_third_gives_full_tv():
    back, front = _zone_in(5, 40), _zone_in(110, 40)
    stats = compare_windows([_move(0, back)] * 5, [_move(0, front)] * 5)
    assert stats["tv_dest"] == 1.0


def test_rate_differences_are_absolute():
    z = _zone_in(60, 40)
    before = [_move(0, z, shots=0, loses=2)] * 4
    after = [_move(0, z, shots=1, loses=0)] * 4
    stats = compare_windows(before, after)
    assert stats["d_shot_rate"] == 1.0
    assert stats["d_lose_rate"] == 2.0


def test_window_without_moves_gives_none():
    assert compare_windows([_move(0, 1, c=0.0)] * 3, [_move(0, 1)] * 3) is None


def test_event_index_takes_first_possession_at_or_after():
    assert event_index([3, 5, 9], 5) == 1
    assert event_index([3, 5, 9], 6) == 2
    assert event_index([3, 5, 9], 10) is None


def test_window_stat_rejects_windows_crossing_half_time():
    z = _zone_in(60, 40)
    poss = [(i, 1 if i < 6 else 2, _move(0, z)) for i in range(12)]
    assert window_stat(poss, 6, 3) is None      # before is 1st half, after is 2nd
    assert window_stat(poss, 4, 3) is None      # after-window spans the break
    assert window_stat(poss, 3, 3) is not None  # 0-2 | 3-5, all first half
    assert window_stat(poss, 2, 3) is None      # not enough possessions before


def test_auc_is_half_for_identical_and_one_for_separated():
    assert auc_vs_baseline(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5
    assert auc_vs_baseline(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0
