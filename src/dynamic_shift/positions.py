"""Identity-aware position change around a moment (v2, see README).

Every earlier check summarised *how the ball moved* or an anonymous
shape. A formation change is specific players taking up new positions, and
events carry player ids and locations. This module compares, for the
players present in both windows, where each one touched the ball before
vs after a moment.

Coordinates are StatsBomb's, in the event team's own frame (it attacks
toward x=120 in both halves), so no mirroring is needed.

Time axis: `elapsed()` puts periods 1 and 2 on one continuous clock in
seconds (period 2 starts where period 1's last event ended), so a window
can straddle half-time without first-half stoppage time colliding with
second-half minute 45-47.

Features (arrays in, floats out; no lake needed to test):
  disp_rel   mean distance a continuing player's centroid moved, after
             removing the whole team's average shift (so the block
             pushing up or dropping off does not count)
  order_flip share of player pairs whose left/right or ahead/behind
             ordering reversed (relative structure, block-shift proof)
"""
from __future__ import annotations

import numpy as np

WINDOW_SECONDS = 600.0
MIN_TOUCHES = 5
MIN_PLAYERS = 6
FEATURE_NAMES = ("disp_rel", "order_flip")


def elapsed(period: np.ndarray, period_seconds: np.ndarray) -> np.ndarray:
    """Continuous seconds across periods 1 and 2. Other periods -> NaN."""
    period = np.asarray(period)
    ps = np.asarray(period_seconds, dtype=float)
    p1 = ps[period == 1]
    p1_end = float(p1.max()) if len(p1) else 0.0
    out = np.full(len(ps), np.nan)
    out[period == 1] = ps[period == 1]
    out[period == 2] = p1_end + ps[period == 2]
    return out


def centroids(t, player, x, y, lo: float, hi: float,
              min_touches: int = MIN_TOUCHES) -> dict:
    """player -> (mean x, mean y) over touches with lo <= t < hi."""
    t, player = np.asarray(t, float), np.asarray(player)
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (t >= lo) & (t < hi) & np.isfinite(x) & np.isfinite(y)
    out = {}
    for p in np.unique(player[keep]):
        m = keep & (player == p)
        if m.sum() >= min_touches:
            out[p] = (float(x[m].mean()), float(y[m].mean()))
    return out


def change_features(before: dict, after: dict, min_players: int = MIN_PLAYERS) -> dict | None:
    """Compare two `centroids` dicts on the players present in both."""
    common = sorted(set(before) & set(after))
    if len(common) < min_players:
        return None
    b = np.array([before[p] for p in common])
    a = np.array([after[p] for p in common])
    delta = a - b
    rel = delta - delta.mean(axis=0)
    disp_rel = float(np.linalg.norm(rel, axis=1).mean())

    flips, pairs = 0, 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            for axis in (0, 1):
                db, da = b[i, axis] - b[j, axis], a[i, axis] - a[j, axis]
                pairs += 1
                flips += int(db * da < 0)
    return {"disp_rel": disp_rel, "order_flip": flips / pairs}


def window_change(t, player, x, y, moment: float, end: float,
                  window: float = WINDOW_SECONDS) -> dict | None:
    """Features for the `window` seconds before vs after `moment`.

    Both windows must be complete (inside [0, end]); this is the same rule
    for real events and random moments.
    """
    if moment - window < 0 or moment + window > end:
        return None
    before = centroids(t, player, x, y, moment - window, moment)
    after = centroids(t, player, x, y, moment, moment + window)
    return change_features(before, after)
