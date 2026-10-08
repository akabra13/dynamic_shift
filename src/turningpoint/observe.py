"""Possession -> observation vector, for feeding monitor.py.

One observation is a possession's zone-transition counts, flattened over
pitchgraph's TACTICAL grid vocabulary: n=30 zones x (n move destinations
+ shot + lose) = 960 categories, zone-major then outcome. This is the
exact (start zone, outcome) layout pitchgraph.chains.counts.outcome_table
counts per match -- just grouped by possession here instead of by match
-- which means a team's already-fitted v3 shrunk chain
(pitchgraph.chains.team.Chain.P) is directly usable as monitor.py's
Dirichlet prior: flatten chain.P * kappa in this same category order,
and there is no leakage because that chain was fit on other matches.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pitchgraph.chains.grid import TACTICAL, Grid
from pitchgraph.chains.team import Chain
from pitchgraph.value.markov import classify_actions


def categories(grid: Grid = TACTICAL) -> list[str]:
    """Flat (start zone, outcome) vocabulary, zone-major then outcome --
    the same order `chain_prior_alpha` and `match_possession_counts` use,
    and the order monitor.py's `prior_alpha` must be given in."""
    n = grid.n
    labels = []
    for z0 in range(n):
        labels += [f"{z0}->{z1}" for z1 in range(n)]
        labels += [f"{z0}->shot", f"{z0}->lose"]
    return labels


def chain_prior_alpha(chain: Chain, kappa: float) -> np.ndarray:
    """A team's shrunk chain as a Dirichlet prior over `categories(chain.grid)`.

    `chain.P` rows already sum to 1, so scaling by `kappa` gives each zone
    `kappa` pseudo-actions worth of prior pseudo-counts -- the same kappa
    units pitchgraph.chains.team already uses for shrinkage.
    """
    return (chain.P * kappa).reshape(-1)


def match_possession_counts(events: pl.LazyFrame, team_id: int,
                             grid: Grid = TACTICAL) -> list[tuple[int, np.ndarray]]:
    """One match's possessions for `team_id`, in chronological order, as
    (possession_id, flattened category-count vector) pairs.

    Only open-play actions count (set-piece deliveries excluded), matching
    how pitchgraph.chains.counts.outcome_table builds a team's chain --
    so a possession's counts live in the same space as the prior.
    """
    n = grid.n
    act = classify_actions(events, open_play_only=True).filter(pl.col("team_id") == team_id)
    out = (pl.when(pl.col("kind") == "move").then(grid.index_expr(pl.col("end_x"), pl.col("end_y")))
             .when(pl.col("kind") == "shot").then(pl.lit(n))
             .otherwise(pl.lit(n + 1)))
    rows = (act.with_columns(z0=grid.index_expr(pl.col("x"), pl.col("y")), out=out.cast(pl.Int32))
               .select("possession", "idx", "z0", "out")
               .collect()
               .sort(["possession", "idx"]))

    k = n + 2
    result = []
    for (poss,), group in rows.group_by(["possession"], maintain_order=True):
        counts = np.zeros(n * k)
        idx = group["z0"].to_numpy() * k + group["out"].to_numpy()
        np.add.at(counts, idx, 1.0)
        result.append((int(poss), counts))
    return result


def possession_minutes(events: pl.LazyFrame, team_id: int) -> dict[int, float]:
    """Each of `team_id`'s possessions' start minute (its first open-play
    action's minute + second/60), for mapping a ground-truth event's clock
    time onto a possession index in `match_possession_counts`'s output."""
    act = classify_actions(events, open_play_only=True).filter(pl.col("team_id") == team_id)
    first = (act.sort("idx")
                .group_by("possession", maintain_order=True)
                .agg(pl.first("minute").alias("minute"), pl.first("second").alias("second"))
                .collect())
    return {int(r["possession"]): r["minute"] + r["second"] / 60.0
            for r in first.iter_rows(named=True)}
