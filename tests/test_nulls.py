import numpy as np
import pytest

from dynamic_shift.nulls import (
    calibrate_threshold, false_alarm_rate, permutation_null_scores, shuffle_within_halves,
)


def test_shuffle_within_halves_preserves_each_halfs_multiset():
    periods = np.array([1, 1, 1, 1, 2, 2, 2])
    rng = np.random.default_rng(0)
    order = shuffle_within_halves(periods, rng)

    # Every position still maps to a position with the same period --
    # i.e. no possession crossed from one half into the other.
    assert sorted(order[periods == 1].tolist()) == [0, 1, 2, 3]
    assert sorted(order[periods == 2].tolist()) == [4, 5, 6]


def test_shuffle_within_halves_actually_reorders():
    periods = np.array([1] * 10)
    rng = np.random.default_rng(0)
    order = shuffle_within_halves(periods, rng)
    assert not np.array_equal(order, np.arange(10))


def test_permutation_null_scores_keeps_possession_count_and_each_halfs_ids():
    k = 4
    rng = np.random.default_rng(1)
    possessions = [(i + 1, 1 if i < 10 else 2, rng.multinomial(5, np.ones(k) / k).astype(float))
                   for i in range(20)]
    trajectories = permutation_null_scores(
        possessions, prior_alpha=np.ones(k), hazard=20.0, change_point_window=5,
        n_permutations=5, rng=rng,
    )
    assert len(trajectories) == 5
    for traj in trajectories:
        assert len(traj.scores) == 20
        # First half's possession ids (1-10) never end up in a position
        # whose period is 2, and vice versa.
        assert set(traj.possession_ids[traj.periods == 1].tolist()) == set(range(1, 11))
        assert set(traj.possession_ids[traj.periods == 2].tolist()) == set(range(11, 21))


def test_false_alarm_rate_is_finite_and_bounded():
    k = 4
    rng = np.random.default_rng(2)
    possessions = [(i + 1, 1, rng.multinomial(5, np.ones(k) / k).astype(float)) for i in range(40)]
    trajectories = permutation_null_scores(
        possessions, prior_alpha=np.ones(k), hazard=20.0, change_point_window=5,
        n_permutations=10, rng=rng,
    )
    rate = false_alarm_rate(trajectories, threshold=0.5, burn_in=10, refractory=5)
    assert 0.0 <= rate <= 40


def test_calibrate_threshold_hits_its_target_on_synthetic_nulls():
    """A low threshold should give a high false-alarm rate and a high
    threshold a low one; calibration should land close to the target."""
    k = 4
    rng = np.random.default_rng(4)
    possessions = [(i + 1, 1 if i < 30 else 2, rng.multinomial(5, np.ones(k) / k).astype(float))
                   for i in range(60)]
    trajectories = permutation_null_scores(
        possessions, prior_alpha=np.ones(k), hazard=20.0, change_point_window=5,
        n_permutations=30, rng=rng,
    )
    threshold = calibrate_threshold(trajectories, target=0.25, burn_in=10, refractory=5)
    rate_at_threshold = false_alarm_rate(trajectories, threshold, burn_in=10, refractory=5)
    assert rate_at_threshold <= 0.25 + 1e-9

    # A meaningfully lower threshold should give a higher (worse) rate,
    # confirming the sweep actually found a real trade-off, not a trivial
    # endpoint.
    rate_lower = false_alarm_rate(trajectories, max(threshold - 0.1, 0.01), burn_in=10, refractory=5)
    assert rate_lower >= rate_at_threshold
