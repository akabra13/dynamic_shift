"""Identity-aware position features and the formation shape vector."""
import numpy as np

from dynamic_shift.formation import balanced_accuracy, parse_formation, shape_vector
from dynamic_shift.positions import centroids, change_features, elapsed, window_change


def _line(players, xs, ys):
    return {p: (x, y) for p, x, y in zip(players, xs, ys)}


def test_elapsed_is_continuous_across_the_break():
    period = np.array([1, 1, 2, 2, 3])
    ps = np.array([100.0, 2800.0, 0.0, 30.0, 5.0])
    e = elapsed(period, ps)
    assert e[1] == 2800 and e[2] == 2800 and e[3] == 2830 and np.isnan(e[4])


def test_whole_block_shift_is_not_a_change():
    ids = list(range(8))
    b = _line(ids, np.linspace(20, 70, 8), np.linspace(10, 70, 8))
    a = {p: (x + 10, y + 3) for p, (x, y) in b.items()}
    f = change_features(b, a)
    assert f["disp_rel"] < 1e-9 and f["order_flip"] == 0


def test_players_swapping_roles_is_a_change():
    ids = list(range(8))
    xs, ys = np.linspace(20, 70, 8), np.linspace(10, 70, 8)
    b = _line(ids, xs, ys)
    a = _line(ids, xs[::-1], ys[::-1])           # everyone takes the opposite role
    f = change_features(b, a)
    assert f["disp_rel"] > 15 and f["order_flip"] == 1.0


def test_too_few_common_players_gives_none():
    b = _line(range(8), np.arange(8.0), np.arange(8.0))
    a = _line(range(5, 13), np.arange(8.0), np.arange(8.0))     # only 3 in common
    assert change_features(b, a) is None


def test_centroids_need_min_touches_and_window():
    t = np.array([1, 2, 3, 4, 5, 6, 100.0])
    cents = centroids(t, np.array([1] * 6 + [2]), np.arange(7.0), np.arange(7.0), 0, 50)
    assert list(cents) == [1] and cents[1] == (2.5, 2.5)


def test_window_change_needs_complete_windows():
    t = np.linspace(0, 1000, 400)
    pl_ = np.tile(np.arange(8), 50)
    x = np.ones(400) * 50
    assert window_change(t, pl_, x, x, moment=100, end=1000, window=600) is None
    assert window_change(t, pl_, x, x, moment=500, end=1100, window=450) is not None


def test_formation_parse_and_shape_vector():
    assert parse_formation(4231) == {"back_line": 4, "forwards": 1}
    assert parse_formation(352) == {"back_line": 3, "forwards": 2}
    assert parse_formation(99) is None and parse_formation(None) is None
    cents = {i: (10.0 * (10 - i), 40.0 + i) for i in range(10)}
    v = shape_vector(cents)
    assert len(v) == 20 and v[0] == 10 and list(v[:10]) == sorted(v[:10])
    assert shape_vector({0: (1, 1)}) is None


def test_window_vector_takes_ten_busiest_players():
    from dynamic_shift.formation import window_vector
    # 12 players; players 10 and 11 have only 3 touches, the rest 5
    player = np.concatenate([np.repeat(np.arange(10), 5), np.repeat([10, 11], 3)])
    t = np.linspace(0, 99, len(player))
    x = player.astype(float) * 10
    v = window_vector(t, player, x, np.full(len(player), 40.0), 0, 100)
    assert v is not None and sorted(v[:10]) == list(np.arange(10) * 10.0)
    assert window_vector(t, player, x, x, 0, 30) is None      # too few players in window


def test_balanced_accuracy_penalises_majority_guessing():
    y = [3, 4, 4, 4, 4, 4, 4, 4, 4, 5]
    assert balanced_accuracy(y, [4] * 10) == 1 / 3
    assert balanced_accuracy(y, y) == 1.0
