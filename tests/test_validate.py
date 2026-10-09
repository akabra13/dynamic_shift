"""End-to-end: leave-one-match-out prior + detection, on a hand-constructed
scenario where the held-out match has a clean injected shift partway through."""
import polars as pl
import pytest

from pitchgraph.chains.counts import CountStore
from pitchgraph.chains.grid import Grid

from dynamic_shift.validate import detect_within, leave_one_out_prior

GRID = Grid.uniform(2, 1)  # zone 0: x<60, zone 1: x>=60


def _ev(match_id, idx, typ, x, y=40.0, end_x=None, outcome=None, xg=None,
        poss=1, minute=0, second=0.0, team=1):
    return {"match_id": match_id, "idx": idx, "period": 1, "type": typ,
            "team_id": team, "possession": poss, "possession_team_id": team,
            "home_team_id": 1, "away_team_id": 2, "x": x, "y": y,
            "end_x": end_x, "end_y": y if end_x is not None else None,
            "outcome": outcome, "xg": xg, "sub_type": None,
            "minute": minute, "second": second}


def _stable_match(match_id, n_possessions=20):
    """Every possession: one pass that stays in zone 0 (self-loop)."""
    return [_ev(match_id, idx=p, typ="Pass", x=10.0, end_x=20.0, poss=p, minute=p)
            for p in range(1, n_possessions + 1)]


def _held_out_match_with_injected_shift(match_id, n_before=10, n_after=10):
    """First n_before possessions: same stable zone0->zone0 pattern. Then
    an abrupt switch to shooting from zone 0 every possession."""
    rows = [_ev(match_id, idx=p, typ="Pass", x=10.0, end_x=20.0, poss=p, minute=p)
            for p in range(1, n_before + 1)]
    rows += [_ev(match_id, idx=p, typ="Shot", x=10.0, xg=0.3, poss=p, minute=p)
             for p in range(n_before + 1, n_before + n_after + 1)]
    return rows


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None).lazy()


def test_leave_one_out_prior_excludes_the_held_out_match():
    all_rows = _stable_match(1) + _stable_match(2) + _held_out_match_with_injected_shift(3)
    store = CountStore.from_events(_frame(all_rows), grid=GRID)
    chain = leave_one_out_prior(store, team_id=1, exclude_match=3, kappa=1.0)
    # The prior is built only from matches 1 and 2, which never shoot, so
    # it should reflect "always moves zone0->zone0", not the held-out
    # match's shooting.
    assert chain.move[0, 0] > 0.99
    assert chain.shoot[0] < 0.01


def test_detect_within_finds_the_injected_shift():
    all_rows = _stable_match(1) + _stable_match(2) + _held_out_match_with_injected_shift(3)
    store = CountStore.from_events(_frame(all_rows), grid=GRID)
    chain = leave_one_out_prior(store, team_id=1, exclude_match=3, kappa=1.0)

    held_out_events = _frame(_held_out_match_with_injected_shift(3))
    result = detect_within(held_out_events, team_id=1, chain=chain, around_minute=11.0,
                            prior_strength=5.0, hazard=50.0, threshold=0.15,
                            within_possessions=10, grid=GRID)
    assert result.detected is True
    assert result.latency is not None and result.latency <= 5


def test_detect_within_does_not_fire_on_a_stable_stretch():
    """A negative control: checking a point in an otherwise-stable match
    should not detect anything, matching the design doc's requirement
    that negative controls flag at about the base rate, not constantly."""
    all_rows = _stable_match(1) + _stable_match(2) + _stable_match(3, n_possessions=20)
    store = CountStore.from_events(_frame(all_rows), grid=GRID)
    chain = leave_one_out_prior(store, team_id=1, exclude_match=3, kappa=1.0)

    held_out_events = _frame(_stable_match(3, n_possessions=20))
    result = detect_within(held_out_events, team_id=1, chain=chain, around_minute=11.0,
                            prior_strength=5.0, hazard=50.0, threshold=0.15,
                            within_possessions=9, grid=GRID)
    assert result.detected is False
