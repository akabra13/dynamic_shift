"""Candidate-event extraction and naive proximity ranking, on synthetic events."""
import json

import polars as pl
import pytest

from turningpoint.attribute import candidate_events, rank_candidate_causes


def _ev(typ, minute, second=0.0, team="Home", team_id=1, player=None,
        outcome=None, extra=None):
    return {"type": typ, "team": team, "team_id": team_id, "player": player,
            "minute": minute, "second": second, "outcome": outcome,
            "extra": json.dumps(extra or {})}


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None).lazy()


def test_candidate_events_filters_by_window_and_type():
    ev = _frame([
        _ev("Pass", 40),                                   # not a candidate type
        _ev("Substitution", 58, player="A",
            extra={"substitution": {"replacement": {"name": "B"}}}),
        _ev("Foul Committed", 61, extra={"foul_committed": {"card": {"name": "Yellow Card"}}}),
        _ev("Shot", 62, outcome="Goal"),
        _ev("Substitution", 20),                            # outside the window
    ])
    out = candidate_events(ev, around_minute=63.0, window_minutes=15.0)
    types = [c["type"] for c in out]
    assert types == ["Substitution", "Foul Committed", "Shot"]
    assert out[0]["description"] == "sub: A off, B on"
    assert out[1]["description"] == "Yellow Card"
    assert out[2]["description"] == "goal"


def test_candidate_events_includes_both_teams():
    ev = _frame([
        _ev("Substitution", 55, team="Home", team_id=1),
        _ev("Substitution", 56, team="Away", team_id=2),
    ])
    out = candidate_events(ev, around_minute=60.0)
    assert {c["team_id"] for c in out} == {1, 2}


def test_rank_candidate_causes_orders_by_proximity():
    candidates = [{"minute": 40.0, "type": "Substitution"},
                  {"minute": 61.5, "type": "Foul Committed"},
                  {"minute": 58.0, "type": "Shot"}]
    ranked = rank_candidate_causes(around_minute=63.0, candidates=candidates)
    assert [c["type"] for c in ranked] == ["Foul Committed", "Shot", "Substitution"]
