import numpy as np

from dynamic_shift.nulls import permutation_false_alarm_rate


def test_false_alarm_rate_is_finite_and_bounded():
    rng = np.random.default_rng(3)
    k = 4
    seq = rng.multinomial(5, np.ones(k) / k, size=40)

    rate = permutation_false_alarm_rate(
        sequences=[seq],
        prior_alpha=np.ones(k),
        hazard=20.0,
        threshold=0.5,
        within_last=5,
        n_permutations=10,
        rng=rng,
    )
    assert 0.0 <= rate <= len(seq)
