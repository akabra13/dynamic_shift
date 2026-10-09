"""Permutation null for calibrating the change-point threshold.

Shuffles a team-match's possessions within each half (content preserved,
timing destroyed, never crossing half-time) and reruns the detector --
in data with no real tactical change, the detector should still flag
roughly the calibrated false-alarm rate, not zero and not constantly.

The detector runs once per permutation and keeps the full score
trajectory (`validate.MatchScores`); sweeping the alarm threshold is
then just re-running `validate.alarms` over the already-computed scores,
which is cheap -- the expensive part (the detector pass) happens exactly
`n_permutations` times, not once per threshold tried.
"""
from __future__ import annotations

import numpy as np

from dynamic_shift.monitor import DirichletMultinomialBOCPD
from dynamic_shift.validate import MatchScores, alarms


def shuffle_within_halves(periods: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of positions 0..len(periods)-1 that reorders possessions
    only within each distinct value of `periods` (never across halves)."""
    order = np.arange(len(periods))
    for period in np.unique(periods):
        idx = np.flatnonzero(periods == period)
        order[idx] = rng.permutation(idx)
    return order


def null_trajectory(possessions: list[tuple[int, int, np.ndarray]], prior_alpha: np.ndarray,
                     hazard: float, change_point_window: int,
                     rng: np.random.Generator) -> MatchScores:
    """One shuffled-within-halves replay of `possessions`
    (possession_id, period, counts triples, as from
    observe.match_possession_counts for one team-match), scored once."""
    poss_ids = np.array([p[0] for p in possessions])
    periods = np.array([p[1] for p in possessions])
    counts = [p[2] for p in possessions]

    order = shuffle_within_halves(periods, rng)
    detector = DirichletMultinomialBOCPD(prior_alpha, hazard=hazard)
    scores = np.empty(len(order))
    for i, j in enumerate(order):
        detector.update(counts[j])
        scores[i] = detector.change_point_probability(change_point_window)
    return MatchScores(poss_ids[order], periods[order], scores)


def permutation_null_scores(possessions: list[tuple[int, int, np.ndarray]], prior_alpha: np.ndarray,
                             hazard: float, change_point_window: int, n_permutations: int = 20,
                             rng: np.random.Generator | None = None) -> list[MatchScores]:
    """`n_permutations` null score trajectories for one team-match."""
    rng = rng or np.random.default_rng()
    return [null_trajectory(possessions, prior_alpha, hazard, change_point_window, rng)
            for _ in range(n_permutations)]


def false_alarm_rate(null_trajectories: list[MatchScores], threshold: float,
                      burn_in: int = 10, refractory: int = 5) -> float:
    """Mean number of distinct alarms per permutation at `threshold` --
    reuses validate.alarms's burn-in/refractory logic, so the null is
    scored the same way real detections are (one alarm per sustained
    crossing, not one per possession above threshold)."""
    if not null_trajectories:
        return 0.0
    counts = [len(alarms(traj, threshold, burn_in, refractory)) for traj in null_trajectories]
    return float(np.mean(counts))


def calibrate_threshold(null_trajectories: list[MatchScores], target: float = 0.25,
                         burn_in: int = 10, refractory: int = 5,
                         grid: np.ndarray | None = None) -> float:
    """The lowest threshold in `grid` whose false-alarm rate across
    `null_trajectories` is still <= `target` -- the most sensitive
    threshold that stays within the false-alarm budget. Falls back to the
    highest (most conservative) grid value if nothing meets the target."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    candidates = [float(t) for t in sorted(grid)
                  if false_alarm_rate(null_trajectories, float(t), burn_in, refractory) <= target]
    return candidates[0] if candidates else float(max(grid))
