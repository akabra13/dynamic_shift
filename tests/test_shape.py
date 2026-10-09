"""Freeze-frame team shape, on synthetic frames."""
import numpy as np
import polars as pl

from dynamic_shift.shape import shape_change, team_shape_rows

T, OTHER = 1, 2


def _make(n_events, line_x_by_event, event_team=T, poss_team=OTHER, period=1):
    """Event i at period_seconds=i*10. Team T has 7 outfield players spread in y at
    x=line_x_by_event(i); event_team decides the frame of reference."""
    ev, fr = [], []
    for i in range(n_events):
        eid = f"e{i}"
        ev.append({"match_id": 1, "event_id": eid, "team_id": event_team,
                   "possession_team_id": poss_team, "period": period,
                   "period_seconds": i * 10.0, "x": 60.0, "y": 40.0})
        for j in range(7):
            x = line_x_by_event(i)
            y = 10.0 + 10.0 * j
            if event_team != T:      # frame of the other team: mirror
                x, y = 120.0 - x, 80.0 - y
            fr.append({"match_id": 1, "event_id": eid, "teammate": event_team == T,
                       "keeper": False, "x": x, "y": y})
    return pl.DataFrame(fr), pl.DataFrame(ev)


def test_shape_rows_mirror_to_team_attack_direction():
    f_own, e_own = _make(3, lambda i: 40.0, event_team=T)
    f_opp, e_opp = _make(3, lambda i: 40.0, event_team=OTHER)
    a = team_shape_rows(f_own, e_own, T)
    b = team_shape_rows(f_opp, e_opp, T)
    assert np.allclose(a["depth"].to_numpy(), 40.0)
    assert np.allclose(b["depth"].to_numpy(), 40.0)   # mirrored back
    assert np.allclose(a["width"].to_numpy(), b["width"].to_numpy())


def test_keepers_and_other_team_excluded():
    f, e = _make(2, lambda i: 40.0)
    extra = pl.DataFrame([{"match_id": 1, "event_id": "e0", "teammate": True,
                           "keeper": True, "x": 2.0, "y": 40.0},
                          {"match_id": 1, "event_id": "e0", "teammate": False,
                           "keeper": False, "x": 100.0, "y": 40.0}])
    rows = team_shape_rows(pl.concat([f, extra]), e, T)
    assert np.allclose(rows["depth"].to_numpy(), 40.0)


def test_oop_flag_follows_possession_team():
    f, e = _make(2, lambda i: 40.0, poss_team=OTHER)
    assert team_shape_rows(f, e, T)["oop"].all()
    f, e = _make(2, lambda i: 40.0, poss_team=T)
    assert not team_shape_rows(f, e, T)["oop"].any()


def test_shape_change_sees_a_line_drop_and_nothing_when_steady():
    # 100 events, 10s apart; line steps from x=50 to x=35 at t=500
    f, e = _make(100, lambda i: 50.0 if i < 50 else 35.0)
    rows = team_shape_rows(f, e, T)
    step = shape_change(rows, 1, 500.0, window_s=400.0)
    assert abs(step["oop_depth"] - 15.0) < 1e-9
    f, e = _make(100, lambda i: 50.0)
    flat = shape_change(team_shape_rows(f, e, T), 1, 500.0, window_s=400.0)
    assert flat["oop_depth"] == 0.0


def test_too_few_frames_gives_no_entry():
    f, e = _make(10, lambda i: 40.0)
    assert shape_change(team_shape_rows(f, e, T), 1, 50.0, window_s=40.0) == {}
