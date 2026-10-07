"""Attribute a detected change to candidate preceding events.

Candidates: subs on both sides, cards, goals/score state, tactical
shifts with formation, half-time, the opponent's own changes -- ranked
by likelihood ratios learned in validation. Always reported as "likely
related to", never "caused by" (guards against the "opponent blocked
the #6" causal trap). Not yet implemented: needs StatsBomb event data
and the validation suite's learned likelihood ratios, neither of which
exist until notebooks/01_validate.ipynb has run against the lake.
"""
from __future__ import annotations


def rank_candidate_causes(change_point, candidate_events) -> list[tuple[object, float]]:
    """Return candidate events ranked by likelihood ratio, most likely first."""
    raise NotImplementedError(
        "Needs learned likelihood ratios from notebooks/01_validate.ipynb."
    )
