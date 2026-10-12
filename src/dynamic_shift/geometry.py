"""Pass geometry and passing-network structure for a window of possessions.

Both take plain arrays so they are testable without a lake. Coordinates
are StatsBomb's, in the passing team's own frame (attacking toward x=120).

pass_geometry(arr): arr is (N, 4) = x, y, end_x, end_y of passes.
  len_mean, len_std   pass length
  fwd                 mean cos(angle): +1 straight upfield, -1 straight back
  lateral             mean |sin(angle)|: how sideways the passing is
  concentration       |mean resultant vector| of angles: 0 = all directions
                      equally, 1 = every pass the same direction
  long_frac           share of passes of 30+ units

network_features(src, dst): player ids of completed passes.
  entropy             Shannon entropy of the directed edge-weight distribution
  algebraic_conn      2nd-smallest eigenvalue of the symmetrised normalised
                      Laplacian (how well-knit the passing graph is)
  clustering          mean weighted clustering coefficient
  centralisation      largest node's share of total pass involvement
"""
from __future__ import annotations

import networkx as nx
import numpy as np

GEOMETRY_NAMES = ("len_mean", "len_std", "fwd", "lateral", "concentration", "long_frac")
NETWORK_NAMES = ("entropy", "algebraic_conn", "clustering", "centralisation")
MIN_PASSES = 8
LONG_PASS = 30.0


def pass_geometry(arr: np.ndarray) -> dict | None:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or len(arr) < MIN_PASSES:
        return None
    dx, dy = arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]
    length = np.hypot(dx, dy)
    ok = length > 1e-9
    if ok.sum() < MIN_PASSES:
        return None
    dx, dy, length = dx[ok], dy[ok], length[ok]
    cos, sin = dx / length, dy / length
    return {
        "len_mean": float(length.mean()),
        "len_std": float(length.std()),
        "fwd": float(cos.mean()),
        "lateral": float(np.abs(sin).mean()),
        "concentration": float(np.hypot(cos.mean(), sin.mean())),
        "long_frac": float((length >= LONG_PASS).mean()),
    }


def network_features(src: np.ndarray, dst: np.ndarray) -> dict | None:
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) < MIN_PASSES:
        return None
    nodes = np.unique(np.concatenate([src, dst]))
    if len(nodes) < 4:
        return None
    index = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for s, d in zip(src, dst):
        if s != d:
            W[index[s], index[d]] += 1.0
    total = W.sum()
    if total <= 0:
        return None

    p = W[W > 0] / total
    entropy = float(-(p * np.log(p)).sum())

    S = W + W.T
    strength = S.sum(axis=1)
    keep = strength > 0
    if keep.sum() < 3:
        return None
    Sk, sk = S[np.ix_(keep, keep)], strength[keep]
    inv = 1.0 / np.sqrt(sk)
    L = np.eye(len(sk)) - inv[:, None] * Sk * inv[None, :]
    eig = np.linalg.eigvalsh(L)
    algebraic = float(eig[1])

    G = nx.from_numpy_array(S)
    clustering = float(np.mean(list(nx.clustering(G, weight="weight").values())))
    centralisation = float(strength.max() / strength.sum())
    return {"entropy": entropy, "algebraic_conn": algebraic,
            "clustering": clustering, "centralisation": centralisation}
