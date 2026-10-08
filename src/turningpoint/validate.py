"""Validate the change-point detector against StatsBomb ground truth.

For a control event (a real formation change, a red card, a same-
formation shift) at some minute in some match, checks whether the
detector flags a change point within a possession window after it --
using a team prior fit on every OTHER match in its competition-season
(leave-one-match-out, so the match being checked never leaks into its
own prior).

Positive controls (real formation changes, red cards) should be
detected; negative controls (same-formation shifts) should be flagged
at roughly the calibrated false-alarm rate (see nulls.py) -- a detector
that flags everything is not a working detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchgraph.chains.counts import CountStore
from pitchgraph.chains.grid import TACTICAL, Grid
from pitchgraph.chains.team import Chain, league_chain, shrunk_chain

from turningpoint.monitor import DirichletMultinomialBOCPD
from turningpoint.observe import chain_prior_alpha, match_possession_counts, possession_minutes


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
class DetectionResult:
    detected: bool
    latency: int | None  # possessions from the event to detection, if detected


def detect_within(events, team_id: int, chain: Chain, around_minute: float,
                   prior_strength: float = 10.0, hazard: float = 20.0,
                   threshold: float = 0.5, within_possessions: int = 15,
                   change_point_window: int = 5, grid: Grid = TACTICAL) -> DetectionResult:
    """Run the online detector over one match's possessions for `team_id`,
    seeded with `chain` as the prior, and check whether a change point is
    flagged within `within_possessions` possessions at/after `around_minute`.

    `prior_strength` is the Dirichlet prior's total pseudo-count per zone
    (how many pseudo-possessions of "no change yet" evidence it starts
    with) -- a separate knob from `chain`'s own shrinkage kappa, which
    controls how the chain *estimate* itself was built.
    """
    possessions = match_possession_counts(events, team_id, grid)
    minutes = possession_minutes(events, team_id)
    alpha = chain_prior_alpha(chain, kappa=prior_strength)
    detector = DirichletMultinomialBOCPD(alpha, hazard=hazard)

    start_idx = None
    for i, (poss_id, counts) in enumerate(possessions):
        detector.update(counts)
        if start_idx is None and minutes.get(poss_id, -1.0) >= around_minute:
            start_idx = i
        if start_idx is None:
            continue
        if detector.change_point_probability(change_point_window) > threshold:
            return DetectionResult(detected=True, latency=i - start_idx)
        if i - start_idx >= within_possessions:
            break
    return DetectionResult(detected=False, latency=None)
