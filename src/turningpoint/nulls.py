"""Permutation null for calibrating the change-point threshold.

Shuffles possessions within each half (content preserved, timing
destroyed), reruns the detector, and reports the false-alarm rate at a
given (posterior-probability, effect-size) threshold pair. Needs a real
match's possession sequence from the lake to run against -- see
observe.py.
"""
from __future__ import annotations

import numpy as np

from turningpoint.monitor import DirichletMultinomialBOCPD


def permutation_false_alarm_rate(
    sequences: list[np.ndarray],
    prior_alpha: np.ndarray,
    hazard: float,
    threshold: float,
    within_last: int,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean false alarms per sequence at `threshold`, over `n_permutations`
    shuffles of each sequence's possession order. Half-preserving shuffles
    are the caller's job: pass each half as a separate sequence so a
    shuffle never mixes across half-time."""
    rng = rng or np.random.default_rng()
    counts = []
    for seq in sequences:
        for _ in range(n_permutations):
            shuffled = rng.permutation(seq)
            detector = DirichletMultinomialBOCPD(prior_alpha=prior_alpha, hazard=hazard)
            count = 0
            for obs in shuffled:
                detector.update(obs)
                if detector.change_point_probability(within_last) > threshold:
                    count += 1
            counts.append(count)
    return float(np.mean(counts)) if counts else 0.0
