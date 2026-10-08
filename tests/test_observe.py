"""Possession -> observation vector, checked against the TACTICAL grid's
own (start zone, outcome) vocabulary on hand-traceable data."""
import numpy as np
import polars as pl
import pytest

from pitchgraph.chains.grid import Grid
from pitchgraph.chains.team import Chain

from turningpoint.observe import categories, chain_prior_alpha, match_possession_counts


def _ev(idx, typ, x, y=40.0, end_x=None, outcome=None, xg=None, sub_type=None,
        team=1, poss=1, pattern="Regular Play"):
    return {"match_id": 1, "event_id": f"e{poss}-{idx}", "idx": idx, "period": 1,
            "period_seconds": float(idx), "type": typ, "team_id": team,
            "possession": poss, "possession_team_id": team, "play_pattern": pattern,
            "x": x, "y": y, "end_x": end_x, "end_y": y if end_x is not None else None,
            "outcome": outcome, "xg": xg, "sub_type": sub_type}


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None).lazy()


def _two_zone_grid() -> Grid:
    return Grid.uniform(2, 1)  # zone 0: x<60, zone 1: x>=60


def test_categories_length_matches_n_times_outcomes():
    grid = _two_zone_grid()
    cats = categories(grid)
    assert len(cats) == grid.n * (grid.n + 2)
    assert cats[0] == "0->0"
    assert cats[grid.n + 1] == "0->lose"


def test_chain_prior_alpha_flattens_in_the_same_order_as_categories():
    grid = _two_zone_grid()
    P = np.array([[0.1, 0.5, 0.2, 0.2],
                  [0.0, 0.0, 0.3, 0.7]])
    chain = Chain(P, np.array([0.1, 0.2]), np.array([1.0, 0.0]), grid)
    alpha = chain_prior_alpha(chain, kappa=10.0)
    cats = categories(grid)
    assert len(alpha) == len(cats)
    # zone 0 -> zone 1 is category index 1 (labels: "0->0","0->1","0->shot","0->lose")
    assert cats[1] == "0->1"
    assert alpha[1] == pytest.approx(10.0 * 0.5)
    # zone 1 -> lose is the last category
    assert cats[-1] == "1->lose"
    assert alpha[-1] == pytest.approx(10.0 * 0.7)


def test_match_possession_counts_on_hand_traceable_possessions():
    grid = _two_zone_grid()
    ev = _frame([
        # Possession 1: a move zone0->zone1, then a shot from zone 1.
        _ev(1, "Pass", 30, end_x=90, poss=1),
        _ev(2, "Shot", 100, xg=0.3, outcome="Saved", poss=1),
        # Possession 2 (later idx): a failed pass (turnover) from zone 0.
        _ev(3, "Pass", 20, end_x=90, outcome="Incomplete", poss=2),
    ])
    result = match_possession_counts(ev, team_id=1, grid=grid)
    by_poss = dict(result)
    assert list(by_poss.keys()) == [1, 2]

    k = grid.n + 2  # 4
    counts1 = by_poss[1]
    assert counts1[0 * k + 1] == 1.0   # zone0 -> zone1 move
    assert counts1[1 * k + grid.n] == 1.0  # zone1 -> shot
    assert counts1.sum() == 2.0

    counts2 = by_poss[2]
    assert counts2[0 * k + (grid.n + 1)] == 1.0  # zone0 -> lose (failed pass)
    assert counts2.sum() == 1.0


def test_possession_counts_feed_directly_into_the_bocpd_detector():
    """End-to-end: a team's chain seeds the prior, and a possession's counts
    are a valid observation for DirichletMultinomialBOCPD.update()."""
    from turningpoint.monitor import DirichletMultinomialBOCPD

    grid = _two_zone_grid()
    P = np.array([[0.1, 0.5, 0.2, 0.2],
                  [0.0, 0.0, 0.3, 0.7]])
    chain = Chain(P, np.array([0.1, 0.2]), np.array([1.0, 0.0]), grid)
    alpha = chain_prior_alpha(chain, kappa=5.0)

    ev = _frame([_ev(1, "Pass", 30, end_x=90, poss=1), _ev(2, "Shot", 100, xg=0.3, poss=1)])
    result = match_possession_counts(ev, team_id=1, grid=grid)

    detector = DirichletMultinomialBOCPD(prior_alpha=alpha, hazard=20.0)
    for _, counts in result:
        probs = detector.update(counts)
        assert probs.sum() == pytest.approx(1.0)
