"""Can a formation be recognised from where players touch the ball? (v2 test B)

Every Starting XI and Tactical Shift carries the team's formation, so each
segment between two such events is a free labelled example. The test asks
whether the formation's *shape class* can be read back from the segment's
per-player touch centroids, without using the lineup's position names.

Targets (digits of the formation, e.g. 4231 -> back line 4, forwards 1):
  back_line   defenders: 3, 4 or 5
  forwards    last row: 1, 2 or 3

Features per segment (10 outfield players, goalkeeper excluded): players
sorted by centroid x; their x values, and |y - 40| in that order, so the
vector describes depth layers and width without knowing who is who.
"""
from __future__ import annotations

import numpy as np

N_OUTFIELD = 10


def parse_formation(code) -> dict | None:
    """4231 -> {'back_line': 4, 'forwards': 1}; anything odd -> None."""
    try:
        digits = [int(c) for c in str(int(code))]
    except (TypeError, ValueError):
        return None
    if len(digits) < 3 or sum(digits) != N_OUTFIELD:
        return None
    return {"back_line": digits[0], "forwards": digits[-1]}


def shape_vector(cents: dict) -> np.ndarray | None:
    """Sorted-by-depth x and width vector from {player: (x, y)}, else None
    unless exactly the ten outfield centroids are available."""
    if len(cents) != N_OUTFIELD:
        return None
    pts = np.array(list(cents.values()), dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    return np.concatenate([pts[:, 0], np.abs(pts[:, 1] - 40.0)])


def window_vector(t, player, x, y, lo: float, hi: float,
                  min_touches: int = 3) -> np.ndarray | None:
    """Shape vector for a time window: the 10 players with the most touches
    (at least `min_touches` each) in lo <= t < hi. Goalkeepers must already
    be excluded by the caller."""
    t, player = np.asarray(t, float), np.asarray(player)
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (t >= lo) & (t < hi) & np.isfinite(x) & np.isfinite(y)
    ids, counts = np.unique(player[keep], return_counts=True)
    ok = counts >= min_touches
    if ok.sum() < N_OUTFIELD:
        return None
    top = ids[ok][np.argsort(-counts[ok], kind="stable")[:N_OUTFIELD]]
    cents = {p: (float(x[keep & (player == p)].mean()), float(y[keep & (player == p)].mean()))
             for p in top}
    return shape_vector(cents)


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    return float(np.mean(recalls))
