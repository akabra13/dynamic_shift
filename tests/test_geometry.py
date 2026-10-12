"""Pass geometry, passing-network features, and the match-resampled AUC."""
import numpy as np

from dynamic_shift.geometry import network_features, pass_geometry
from dynamic_shift.signal_check import auc_ci_by_match, window_ids


def test_all_forward_passes_have_full_concentration_and_no_lateral():
    arr = np.array([[10, 40, 25, 40]] * 10, dtype=float)
    g = pass_geometry(arr)
    assert abs(g["fwd"] - 1) < 1e-9 and abs(g["concentration"] - 1) < 1e-9
    assert g["lateral"] == 0 and g["len_mean"] == 15 and g["long_frac"] == 0


def test_mixed_directions_lower_concentration():
    arr = np.array([[10, 40, 25, 40]] * 5 + [[10, 40, 10, 55]] * 5, dtype=float)
    g = pass_geometry(arr)
    assert g["concentration"] < 1 and 0 < g["lateral"] < 1


def test_too_few_passes_gives_none():
    assert pass_geometry(np.zeros((3, 4))) is None
    assert network_features(np.arange(3), np.arange(3) + 1) is None


def test_network_centralised_hub_vs_ring():
    hub_src = np.array([1, 2, 3, 4] * 3 + [0] * 0)
    hub_dst = np.array([0, 0, 0, 0] * 3)
    ring_src = np.array([0, 1, 2, 3] * 3)
    ring_dst = np.array([1, 2, 3, 0] * 3)
    hub, ring = network_features(hub_src, hub_dst), network_features(ring_src, ring_dst)
    assert hub["centralisation"] > ring["centralisation"]
    assert ring["algebraic_conn"] > 0


def test_window_ids_requires_full_single_half_windows():
    poss = [(i, 1 if i < 8 else 2, np.zeros(2)) for i in range(16)]
    assert window_ids(poss, 4, 4) == ([0, 1, 2, 3], [4, 5, 6, 7])
    assert window_ids(poss, 6, 4) is None     # after-window crosses the break
    assert window_ids(poss, 2, 4) is None     # not enough before


def test_auc_ci_separates_shifted_values_and_covers_half_for_identical():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 400)
    bk = np.repeat(np.arange(100), 4)
    same = rng.normal(0, 1, 200)
    sk = np.repeat(np.arange(100, 150), 4)
    auc, lo, hi = auc_ci_by_match(same, sk, base, bk, rng, level=0.99)
    assert lo < 0.5 < hi
    shifted = rng.normal(2, 1, 200)
    auc, lo, hi = auc_ci_by_match(shifted, sk, base, bk, rng, level=0.99)
    assert lo > 0.8
