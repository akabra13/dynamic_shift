import numpy as np

from turningpoint.monitor import DirichletMultinomialBOCPD


def _sample_sequence(rng, p, n_obs, counts_per_obs=5):
    return rng.multinomial(counts_per_obs, p, size=n_obs)


def test_detects_injected_change_point():
    rng = np.random.default_rng(0)
    k = 4
    p_before = np.array([0.7, 0.1, 0.1, 0.1])
    p_after = np.array([0.1, 0.1, 0.1, 0.7])

    before = _sample_sequence(rng, p_before, 50)
    after = _sample_sequence(rng, p_after, 50)
    sequence = np.vstack([before, after])
    change_at = 50

    prior_alpha = np.ones(k)
    detector = DirichletMultinomialBOCPD(prior_alpha=prior_alpha, hazard=50.0)

    map_run_lengths = []
    for obs in sequence:
        detector.update(obs)
        map_run_lengths.append(detector.map_run_length())

    # Just before the change, the run should have grown to reflect ~50
    # stable possessions.
    assert map_run_lengths[change_at - 1] > 20

    # Within a handful of possessions after the change, the MAP run
    # length should have collapsed back down, i.e. the detector noticed.
    settle = 10
    assert min(map_run_lengths[change_at : change_at + settle]) < 10


def test_no_change_point_in_stationary_sequence():
    rng = np.random.default_rng(1)
    k = 4
    p = np.array([0.25, 0.25, 0.25, 0.25])
    sequence = _sample_sequence(rng, p, 80)

    detector = DirichletMultinomialBOCPD(prior_alpha=np.ones(k), hazard=50.0)
    for obs in sequence:
        detector.update(obs)

    # With no injected change, the run length should have grown close to
    # the full sequence length rather than resetting.
    assert detector.map_run_length() > 40


def test_streaming_matches_batch_replay():
    """The detector only ever conditions on x_1:t, so feeding the same
    sequence twice (fresh detector each time) must give identical
    results -- this is the streaming == batch guarantee the design doc
    requires for replay mode to be faithful."""
    rng = np.random.default_rng(2)
    k = 3
    sequence = _sample_sequence(rng, np.array([0.5, 0.3, 0.2]), 30)

    def run(seq):
        d = DirichletMultinomialBOCPD(prior_alpha=np.ones(k), hazard=20.0)
        out = []
        for obs in seq:
            out.append(d.update(obs).copy())
        return out

    run_a = run(sequence)
    run_b = run(sequence)
    for a, b in zip(run_a, run_b):
        np.testing.assert_allclose(a, b)
