"""Ground-truth controls for validating the change-point detector.

StatsBomb's `tactics` payload (a Starting XI/Tactical Shift event's
formation + lineup) is what lets a "Tactical Shift" be classified as a
real formation change or not -- only possible since pitchgraph's ingest
stopped dropping that field (see pitchgraph's ingest.py history). Per
the design doc, about 30% of Tactical Shift events are real formation
changes; the rest (same formation, e.g. a tweak within the same shape)
are the negative control.

  positive controls   real formation changes; red cards
  negative controls   same-formation Tactical Shifts

"Like-for-like subs" (also a negative control in the design doc) needs
comparing the outgoing/incoming players' positions, which needs lineup
position data this module doesn't yet use -- left for later.
"""
from __future__ import annotations

import json

import polars as pl

TACTICS_TYPES = ("Starting XI", "Tactical Shift")


def _clock(row: dict) -> float:
    return row["minute"] + row["second"] / 60.0


def tactical_shifts(events: pl.LazyFrame, team_id: int) -> list[dict]:
    """Every Tactical Shift for `team_id`, classified against the formation
    that came before it (its most recent Starting XI or Tactical Shift).

    Starting XI events are excluded from the result (there's no "before"
    to compare the first one against) but still used to seed the
    comparison for the first real Tactical Shift of the match.
    """
    rows = (
        events.filter((pl.col("type").is_in(TACTICS_TYPES)) & (pl.col("team_id") == team_id))
              .select("type", "possession", "minute", "second", "extra")
              .sort(["minute", "second"])
              .collect()
    )
    out = []
    previous_formation = None
    for row in rows.iter_rows(named=True):
        extra = json.loads(row["extra"]) if row["extra"] else {}
        formation = (extra.get("tactics") or {}).get("formation")
        if row["type"] == "Tactical Shift":
            out.append({
                "possession": row["possession"], "minute": _clock(row),
                "formation_before": previous_formation, "formation_after": formation,
                "real_change": previous_formation is not None and formation != previous_formation,
            })
        previous_formation = formation
    return out


def _lineup_positions(extra: dict) -> dict[int, str]:
    """player_id -> position name, from a tactics payload's lineup."""
    lineup = (extra.get("tactics") or {}).get("lineup") or []
    return {p["player"]["id"]: p["position"]["name"] for p in lineup
            if p.get("player") and p.get("position")}


def shift_details(events: pl.LazyFrame, team_id: int) -> list[dict]:
    """Each Tactical Shift for `team_id` with how the lineup actually changed.

    For players on the pitch both before and after: `moved` counts those
    whose position changed (goalkeepers excluded), `continuing` how many
    stayed on. `joined`/`left` count players entering/leaving (subs, red
    cards). A 'formation change' with moved == 0 is a relabel or a pure
    substitution, not players taking up new roles.
    """
    rows = (
        events.filter((pl.col("type").is_in(TACTICS_TYPES)) & (pl.col("team_id") == team_id))
              .select("type", "possession", "minute", "second", "extra")
              .sort(["minute", "second"])
              .collect()
    )
    out = []
    prev_formation, prev_pos = None, None
    for row in rows.iter_rows(named=True):
        extra = json.loads(row["extra"]) if row["extra"] else {}
        formation = (extra.get("tactics") or {}).get("formation")
        pos = _lineup_positions(extra)
        if row["type"] == "Tactical Shift" and prev_pos is not None:
            stay = set(pos) & set(prev_pos)
            moved = [p for p in stay if pos[p] != prev_pos[p]
                     and "Goalkeeper" not in (pos[p], prev_pos[p])]
            out.append({
                "possession": row["possession"], "minute": _clock(row),
                "formation_before": prev_formation, "formation_after": formation,
                "real_change": prev_formation is not None and formation != prev_formation,
                "continuing": len(stay), "moved": len(moved),
                "joined": len(set(pos) - set(prev_pos)), "left": len(set(prev_pos) - set(pos)),
            })
        prev_formation, prev_pos = formation, pos
    return out


def red_cards(events: pl.LazyFrame) -> list[dict]:
    """Red card events (direct or second-yellow) for either team."""
    is_foul = pl.col("type") == "Foul Committed"
    is_bad_behaviour = pl.col("type") == "Bad Behaviour"
    rows = (
        events.filter(is_foul | is_bad_behaviour)
              .select("type", "team_id", "possession", "minute", "second", "extra")
              .sort(["minute", "second"])
              .collect()
    )
    out = []
    for row in rows.iter_rows(named=True):
        extra = json.loads(row["extra"]) if row["extra"] else {}
        key = "foul_committed" if row["type"] == "Foul Committed" else "bad_behaviour"
        card = ((extra.get(key) or {}).get("card") or {}).get("name", "")
        if "Red Card" in card:
            out.append({"possession": row["possession"], "minute": _clock(row),
                        "team_id": row["team_id"], "card": card})
    return out


def had_substitution_within(events: pl.LazyFrame, team_id: int, minute: float,
                             window_minutes: float = 1.0) -> bool:
    """Whether `team_id` made a substitution within `window_minutes` of
    `minute` (design doc uses a 60s window to check the detector can't
    succeed just by spotting subs)."""
    clock = pl.col("minute") + pl.col("second") / 60.0
    rows = (
        events.filter((pl.col("type") == "Substitution") & (pl.col("team_id") == team_id))
              .filter((clock >= minute - window_minutes) & (clock <= minute + window_minutes))
              .select("minute")
              .collect()
    )
    return rows.height > 0
