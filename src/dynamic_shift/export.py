"""Export one match's monitor output as the JSON the Library's match
timeline page consumes: both teams' threat curves, event markers,
change bands, and per-change-point before/after chains + sentences +
candidate causes. Not yet implemented: depends on describe.py and
attribute.py being wired up first.
"""
from __future__ import annotations


def export_match(match_id, home_result, away_result) -> dict:
    raise NotImplementedError("Wire up once describe.py and attribute.py are real.")
