"""Possession -> observation vector, for feeding monitor.py.

Not yet implemented: this needs the actual zone/outcome vocabulary and
a possession's coarse 6x4 zone-graph transition counts, which come from
pitchgraph.data.possessions and pitchgraph.value.markov (zone_expr,
classify_actions) once the Drive lake has been built by
notebooks/01_ingest.ipynb + 02_library.ipynb in the pitchgraph repo.
That Colab run is still pending (see pitchgraph README), so this stays
a stub with the interface monitor.py expects rather than a guess at
data it hasn't seen yet.
"""
from __future__ import annotations

import numpy as np


def possession_to_counts(possession, categories: list[str]) -> np.ndarray:
    """Map one possession's events to a category-count vector.

    `categories` is the fixed vocabulary of (zone, outcome) pairs the
    detector was initialised with (its order must match `prior_alpha`).
    """
    raise NotImplementedError(
        "Needs a possession object from pitchgraph.data.possessions; "
        "wire this up once the lake is built."
    )
