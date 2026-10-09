"""Validate the change-point detector against StatsBomb ground truth.

For a control event (a real formation change, a red card, a same-
formation shift) at some possession in some match, checks whether the
detector flags a change point within a possession window after it --
using a team prior fit on every OTHER match in its competition-season
(leave-one-match-out, so the match being checked never leaks into its
own prior).

Positive controls (real formation changes, red cards) should be
detected; negative controls (same-formation shifts) should be flagged
at roughly the calibrated false-alarm rate (see nulls.py) -- a detector
that flags everything is not a working detector.

`score_match` runs the detector once per team-match and keeps the full
score trajectory; `alarms` turns that into a list of alarm possessions
(with burn-in and a refractory period) once, so checking N control
events in the same match costs one detector pass, not N.

Ground-truth events are aligned by StatsBomb's own possession number,
not clock time: clock minutes aren't unique across halves (first-half
stoppage time and second-half time both read e.g. "46:xx"), which
silently misaligned events near half-time under the old minute-based
join.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchgraph.chains.counts import CountStore
from pitchgraph.chains.grid import TACTICAL, Grid
from pitchgraph.chains.team import Chain, league_chain, shrunk_chain

from dynamic_shift.monitor import DirichletMultinomialBOCPD
from dynamic_shift.observe import chain_prior_alpha, match_possession_counts


def leave_one_out_prior(store: CountStore, team_id: int, exclude_match: int,
                         kappa: float) -> Chain:
    """`team_id`'s shrunk chain, fit on every match in `store` except
    `exclude_match` -- both the team's own counts and the league prior
    it's shrunk toward exclude it, so nothing from that match leaks in."""
    league = league_chain(*store.league_total(exclude_match=exclude_match), store.grid)
    tc = store.team(team_id)
    keep = tc.match_ids != exclude_match
    C, xg, st = tc.total(keep)
    return shrunk_chain(C, xg, st, league, kappa)


@dataclass
class MatchScores:
    """One team-match's change-point score trajectory, in chronological
    (possession) order."""
    possession_ids: np.ndarray   # (T,) StatsBomb possession id at each step
    periods: np.ndarray          # (T,) period (1/2/...) at each step
    scores: np.ndarray           # (T,) P(change within last `window`) at each step


def score_match(events, team_id: int, chain: Chain, prior_strength: float = 10.0,
                 hazard: float = 20.0, change_point_window: int = 5,
                 grid: Grid = TACTICAL) -> MatchScores:
    """Run the online detector once over one match's possessions for
    `team_id`, seeded with `chain` as the prior, and keep the full score
    trajectory (not just a single yes/no) so it can be reused to check
    many control events without rerunning the detector.
    """
    possessions = match_possession_counts(events, team_id, grid)
    alpha = chain_prior_alpha(chain, kappa=prior_strength)
    detector = DirichletMultinomialBOCPD(alpha, hazard=hazard)

    poss_ids = np.empty(len(possessions), dtype=np.int64)
    periods = np.empty(len(possessions), dtype=np.int64)
    scores = np.empty(len(possessions), dtype=np.float64)
    for i, (poss_id, period, counts) in enumerate(possessions):
        detector.update(counts)
        poss_ids[i] = poss_id
        periods[i] = period
        scores[i] = detector.change_point_probability(change_point_window)
    return MatchScores(poss_ids, periods, scores)


def alarms(match_scores: MatchScores, threshold: float, burn_in: int = 10,
           refractory: int = 5) -> list[int]:
    """Indices (positions in `match_scores`, not possession ids) where the
    score first crosses `threshold`, ignoring the match's first `burn_in`
    possessions (the detector has no history yet, so its score there is
    uninformative) and suppressing repeat alarms within `refractory`
    possessions of the last one."""
    out = []
    last = -(refractory + 1)
    for i, s in enumerate(match_scores.scores):
        if i < burn_in:
            continue
        if s > threshold and i - last > refractory:
            out.append(i)
            last = i
    return out


def possession_index(match_scores: MatchScores, possession: int) -> int | None:
    """Index of the first of this team's possessions whose StatsBomb
    possession id is >= `possession` -- aligns a ground-truth event (given
    in StatsBomb's global possession numbering) onto this team's own
    possession-indexed score trajectory. None if the event is after this
    team's last possession in the match."""
    idx = int(np.searchsorted(match_scores.possession_ids, possession, side="left"))
    return idx if idx < len(match_scores.possession_ids) else None


@dataclass
class DetectionResult:
    detected: bool
    latency: int | None  # possessions from the event to detection, if detected


def check_control(match_scores: MatchScores, event_possession: int,
                   alarm_indices: list[int], within_possessions: int = 15) -> DetectionResult:
    """Whether any alarm in `alarm_indices` falls within `within_possessions`
    possessions at/after `event_possession`."""
    start = possession_index(match_scores, event_possession)
    if start is None:
        return DetectionResult(detected=False, latency=None)
    for i in alarm_indices:
        if start <= i < start + within_possessions:
            return DetectionResult(detected=True, latency=i - start)
    return DetectionResult(detected=False, latency=None)
