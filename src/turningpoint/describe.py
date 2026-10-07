"""Before/after chain comparison -> plain-English change sentences.

Not yet implemented: needs a team's actual flow-graph edges (lanes,
press height, threat per possession) on either side of a detected
change point, which means a built possession sequence (see observe.py)
and pitchgraph.chains.team to fit the before/after chains.
"""
from __future__ import annotations


def describe_change(before_chain, after_chain) -> list[str]:
    """Return the biggest edge/lane/press-height/threat shifts as
    sentences, e.g. "From 63', their build-up moved left: 41% of
    entries via the left flank, up from 22%"."""
    raise NotImplementedError(
        "Needs before/after team chains from pitchgraph.chains.team; "
        "wire this up once the lake is built."
    )
