"""Attribute a detected change to candidate preceding events.

Candidates: subs on both sides, cards, goals, half-time, and "Tactical
Shift" events (both teams) -- pulled straight from the lake's events
table, which already has everything these need (see
`candidate_events`). Always reported as "likely related to", never
"caused by": `rank_candidate_causes` currently ranks by temporal
proximity alone, a naive placeholder, not the likelihood ratios the
design calls for -- those need notebooks/01_validate.ipynb's positive/
negative-control calibration (StatsBomb Tactical Shift events), which
doesn't exist yet.

Known data gap: StatsBomb's `tactics` payload (the formation itself) is
dropped by pitchgraph's ingest (it's in ingest.CORE_KEYS, so it's
excluded from both the event's own columns and the `extra` JSON catch-
all). That means a "Tactical Shift" candidate here is "a tactical shift
happened", not "formation changed from X to Y" -- distinguishing real
formation changes from the 70% that aren't (per the design doc) needs
that field, so it needs an ingest change plus a re-ingest, not just
code here.
"""
from __future__ import annotations

import json

import polars as pl

CANDIDATE_TYPES = ("Substitution", "Foul Committed", "Bad Behaviour",
                    "Half Start", "Half End", "Tactical Shift")


def _describe(row: dict) -> str:
    extra = json.loads(row["extra"]) if row.get("extra") else {}
    t = row["type"]
    if t == "Substitution":
        sub = extra.get("substitution") or {}
        repl = (sub.get("replacement") or {}).get("name")
        return f"sub: {row.get('player') or '?'} off" + (f", {repl} on" if repl else "")
    if t in ("Foul Committed", "Bad Behaviour"):
        key = "foul_committed" if t == "Foul Committed" else "bad_behaviour"
        card = (extra.get(key) or {}).get("card", {}).get("name")
        return f"{card}" if card else t
    if t == "Shot":
        return "goal"
    return t


def candidate_events(events: pl.LazyFrame, around_minute: float,
                      window_minutes: float = 15.0) -> list[dict]:
    """Candidate causes for both teams within `window_minutes` before
    `around_minute`, earliest first.

    `events` needs the columns pitchgraph.data.lake.Lake.events() joins
    in (team, player, minute, second, period, type, sub_type, outcome,
    extra), already true of every match in the lake.
    """
    is_candidate = pl.col("type").is_in(CANDIDATE_TYPES)
    is_goal = (pl.col("type") == "Shot") & (pl.col("outcome") == "Goal")
    clock = pl.col("minute") + pl.col("second") / 60.0

    rows = (
        events.filter(is_candidate | is_goal)
              .filter((clock <= around_minute) & (clock >= around_minute - window_minutes))
              .select("type", "team", "team_id", "player", "minute", "second", "extra")
              .sort(["minute", "second"])
              .collect()
    )
    return [
        {"minute": row["minute"] + row["second"] / 60.0, "type": row["type"],
         "team_id": row["team_id"], "team": row["team"], "description": _describe(row)}
        for row in rows.iter_rows(named=True)
    ]


def rank_candidate_causes(around_minute: float, candidates: list[dict]) -> list[dict]:
    """Candidates ranked by temporal proximity to `around_minute`, nearest
    first -- a placeholder ranking. The design calls for likelihood ratios
    learned from positive/negative controls (see module docstring); until
    notebooks/01_validate.ipynb exists, proximity is the only signal here,
    so this should be read as "candidates to check", not a probability."""
    return sorted(candidates, key=lambda c: abs(around_minute - c["minute"]))
