"""Possession -> observation vector, checked against the TACTICAL grid's
own (start zone, outcome) vocabulary on hand-traceable data."""
import numpy as np
import polars as pl
import pytest

from pitchgraph.chains.grid import Grid
from pitchgraph.chains.team import Chain

from dynamic_shift.observe import categories, chain_prior_alpha, match_possession_counts, possession_minutes


def _ev(idx, typ, x, y=40.0, end_x=None, outcome=None, xg=None, sub_type=None,
        team=1, poss=1, pattern="Regular Play", minute=0, second=0.0, period=1):
    return {"match_id": 1, "event_id": f"e{poss}-{idx}", "idx": idx, "period": period,
            "period_seconds": float(idx), "type": typ, "team_id": team,
            "possession": poss, "possession_team_id": team, "play_pattern": pattern,
            "x": x, "y": y, "end_x": end_x, "end_y": y if end_x is not None else None,
            "outcome": outcome, "xg": xg, "sub_type": sub_type,
            "minute": minute, "second": second}


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
    alpha = chain_prior_alpha(chain, kappa=10.0, floor=0.0)
    cats = categories(grid)
    assert len(alpha) == len(cats)
    # zone 0 -> zone 1 is category index 1 (labels: "0->0","0->1","0->shot","0->lose")
    assert cats[1] == "0->1"
    assert alpha[1] == pytest.approx(10.0 * 0.5)
    # zone 1 -> lose is the last category
    assert cats[-1] == "1->lose"
    assert alpha[-1] == pytest.approx(10.0 * 0.7)


def test_chain_prior_alpha_floor_keeps_unseen_moves_possible():
    """A move the prior chain assigns exactly 0 probability to (e.g. zone 1
    -> zone 0, never observed in this hand-built chain) must still have a
    positive alpha after the floor, so a single rare occurrence of it
    doesn't get treated as impossible by monitor.py."""
    grid = _two_zone_grid()
    P = np.array([[0.1, 0.5, 0.2, 0.2],
                  [0.0, 0.0, 0.3, 0.7]])
    chain = Chain(P, np.array([0.1, 0.2]), np.array([1.0, 0.0]), grid)
    alpha = chain_prior_alpha(chain, kappa=10.0, floor=0.01)
    cats = categories(grid)
    assert alpha[cats.index("1->0")] == pytest.approx(0.01)


def test_match_possession_counts_on_hand_traceable_possessions():
    grid = _two_zone_grid()
    ev = _frame([
        # Possession 1 (first half): a move zone0->zone1, then a shot from zone 1.
        _ev(1, "Pass", 30, end_x=90, poss=1, period=1),
        _ev(2, "Shot", 100, xg=0.3, outcome="Saved", poss=1, period=1),
        # Possession 2 (second half, later idx): a failed pass (turnover) from zone 0.
        _ev(3, "Pass", 20, end_x=90, outcome="Incomplete", poss=2, period=2),
    ])
    result = match_possession_counts(ev, team_id=1, grid=grid)
    by_poss = {poss: (period, counts) for poss, period, counts in result}
    assert list(by_poss.keys()) == [1, 2]

    k = grid.n + 2  # 4
    period1, counts1 = by_poss[1]
    assert period1 == 1
    assert counts1[0 * k + 1] == 1.0   # zone0 -> zone1 move
    assert counts1[1 * k + grid.n] == 1.0  # zone1 -> shot
    assert counts1.sum() == 2.0

    period2, counts2 = by_poss[2]
    assert period2 == 2
    assert counts2[0 * k + (grid.n + 1)] == 1.0  # zone0 -> lose (failed pass)
    assert counts2.sum() == 1.0


def test_possession_counts_feed_directly_into_the_bocpd_detector():
    """End-to-end: a team's chain seeds the prior, and a possession's counts
    are a valid observation for DirichletMultinomialBOCPD.update()."""
    from dynamic_shift.monitor import DirichletMultinomialBOCPD

    grid = _two_zone_grid()
    P = np.array([[0.1, 0.5, 0.2, 0.2],
                  [0.0, 0.0, 0.3, 0.7]])
    chain = Chain(P, np.array([0.1, 0.2]), np.array([1.0, 0.0]), grid)
    alpha = chain_prior_alpha(chain, kappa=5.0)

    ev = _frame([_ev(1, "Pass", 30, end_x=90, poss=1), _ev(2, "Shot", 100, xg=0.3, poss=1)])
    result = match_possession_counts(ev, team_id=1, grid=grid)

    detector = DirichletMultinomialBOCPD(prior_alpha=alpha, hazard=20.0)
    for _, _, counts in result:
        probs = detector.update(counts)
        assert probs.sum() == pytest.approx(1.0)


def test_possession_minutes_uses_each_possessions_first_action():
    grid = _two_zone_grid()
    ev = _frame([
        _ev(1, "Pass", 30, end_x=90, poss=1, minute=10, second=0.0),
        _ev(2, "Shot", 100, xg=0.3, poss=1, minute=10, second=30.0),  # same possession, later
        _ev(3, "Pass", 20, end_x=90, poss=2, minute=11, second=15.0),
    ])
    minutes = possession_minutes(ev, team_id=1)
    assert minutes == {1: pytest.approx(10.0), 2: pytest.approx(11.25)}
