"""Ground-truth control extraction, on synthetic events."""
import json

import polars as pl

from dynamic_shift.ground_truth import had_substitution_within, red_cards, tactical_shifts


def _ev(typ, minute, second=0.0, team_id=1, extra=None):
    return {"type": typ, "team_id": team_id, "minute": minute, "second": second,
            "extra": json.dumps(extra or {})}


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None).lazy()


def test_tactical_shift_classified_as_real_change_against_starting_xi():
    ev = _frame([
        _ev("Starting XI", 0, extra={"tactics": {"formation": 433}}),
        _ev("Tactical Shift", 60, extra={"tactics": {"formation": 442}}),
    ])
    out = tactical_shifts(ev, team_id=1)
    assert len(out) == 1
    assert out[0]["formation_before"] == 433
    assert out[0]["formation_after"] == 442
    assert out[0]["real_change"] is True


def test_tactical_shift_classified_as_same_formation():
    ev = _frame([
        _ev("Starting XI", 0, extra={"tactics": {"formation": 433}}),
        _ev("Tactical Shift", 60, extra={"tactics": {"formation": 433}}),
    ])
    out = tactical_shifts(ev, team_id=1)
    assert out[0]["real_change"] is False


def test_red_card_detected_not_yellow():
    ev = _frame([
        _ev("Foul Committed", 30, extra={"foul_committed": {"card": {"name": "Yellow Card"}}}),
        _ev("Foul Committed", 70, extra={"foul_committed": {"card": {"name": "Second Yellow"}}}),
        _ev("Bad Behaviour", 80, extra={"bad_behaviour": {"card": {"name": "Red Card"}}}),
    ])
    out = red_cards(ev)
    assert len(out) == 1
    assert out[0]["minute"] == 80.0


def test_had_substitution_within_window():
    ev = _frame([_ev("Substitution", 58, team_id=1)])
    assert had_substitution_within(ev, team_id=1, minute=58.5, window_minutes=1.0) is True
    assert had_substitution_within(ev, team_id=1, minute=65.0, window_minutes=1.0) is False
    assert had_substitution_within(ev, team_id=2, minute=58.0, window_minutes=1.0) is False
