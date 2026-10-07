"""Bayesian online change-point detection (Adams & MacKay, 2007) over
Dirichlet-multinomial count observations.

Each observation is a vector of category counts -- e.g. flattened
zone-transition counts for one possession. Counts within a run are
modelled as i.i.d. multinomial draws with a Dirichlet prior, so the
predictive probability and the posterior update are closed-form.

Scope note: this flattens the zone graph into one set of categories
rather than modelling each zone row as its own conditional multinomial.
That is the honest scope of what is implemented and tested here with
synthetic data. The per-zone-row conditional version described in the
design doc is the natural refinement once real possession sequences are
flowing from the Drive lake (observe.py), since that needs the v3 team
chain as the prior and the actual zone/outcome vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


def _dirichlet_multinomial_log_pred(alpha: np.ndarray, counts: np.ndarray) -> float:
    """Log predictive probability of `counts` under Dirichlet(alpha).

    This is the Dirichlet-multinomial (multivariate Polya) log-pmf with
    the counts' own multinomial coefficient dropped: it is the same
    constant for every run length at a given t, so it cancels when the
    BOCPD recursion renormalises P(r_t | x_1:t) and can be safely left
    out.
    """
    a0 = alpha.sum()
    n = counts.sum()
    return float(
        gammaln(a0) - gammaln(a0 + n)
        + np.sum(gammaln(alpha + counts) - gammaln(alpha))
    )


@dataclass
class DirichletMultinomialBOCPD:
    """Online change-point detector with a Dirichlet-multinomial observation model.

    Parameters
    ----------
    prior_alpha : array of shape (K,)
        Dirichlet prior over the K observation categories. In PitchGraph
        terms this is the team's own v3 shrunk flow-graph transition
        counts from *other* matches, scaled to a pseudo-count strength --
        using it as the prior (rather than fitting it on this match) is
        what keeps the detector leak-free.
    hazard : float
        Expected run length between change points, in possessions
        (constant-hazard prior: P(change at each step) = 1 / hazard).
    """

    prior_alpha: np.ndarray
    hazard: float = 20.0

    def __post_init__(self) -> None:
        self.prior_alpha = np.asarray(self.prior_alpha, dtype=float)
        self.run_length_probs = np.array([1.0])
        self.run_alphas = [self.prior_alpha.copy()]
        self.t = 0

    def update(self, counts: np.ndarray) -> np.ndarray:
        """Feed one observation (one possession's category counts).

        Returns the posterior run-length distribution P(r_t | x_1:t)
        after this step, index r = possessions since the last change.
        """
        counts = np.asarray(counts, dtype=float)
        h = 1.0 / self.hazard

        log_pred = np.array(
            [_dirichlet_multinomial_log_pred(a, counts) for a in self.run_alphas]
        )
        pred = np.exp(log_pred)
        joint = self.run_length_probs * pred

        growth = joint * (1 - h)
        cp_mass = (joint * h).sum()

        new_run_length_probs = np.concatenate(([cp_mass], growth))
        total = new_run_length_probs.sum()
        if total > 0:
            new_run_length_probs /= total
        else:
            # Numerical underflow (every hypothesis had ~0 likelihood):
            # fall back to "just changed" rather than propagate NaNs.
            new_run_length_probs = np.zeros_like(new_run_length_probs)
            new_run_length_probs[0] = 1.0

        new_run_alphas = [self.prior_alpha.copy()] + [
            a + counts for a in self.run_alphas
        ]

        self.run_length_probs = new_run_length_probs
        self.run_alphas = new_run_alphas
        self.t += 1
        return self.run_length_probs

    def change_point_probability(self, within_last: int) -> float:
        """P(a change point occurred within the last `within_last` possessions)."""
        k = min(within_last, len(self.run_length_probs))
        return float(self.run_length_probs[:k].sum())

    def map_run_length(self) -> int:
        return int(np.argmax(self.run_length_probs))
