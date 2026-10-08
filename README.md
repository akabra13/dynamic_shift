# TurningPoint — live match monitor

*When did this game change, what changed, and what probably caused it?*

Second project in the PitchGraph programme (after [pitchgraph](https://github.com/akabra13/football_graph_predictor) v3, before RoleFit). Depends on pitchgraph's v3 team flow graphs, pinned to commit `3b773a1`.

**Honest scope:** open data is historical, so "live" means the detector is *online* — each decision at time t uses only events up to t. A replay of a past match is then faithful to what a live feed would have seen, and `streaming == batch` is enforced by a test.

## Status (2026-10-08)

The pitchgraph Drive lake and Library are built (all 80 competition-seasons, 13.9M events; 23 competition-seasons in the Library with fitted team chains). Everything below is wired to the real zone vocabulary, not synthetic placeholders.

| Piece | State |
|---|---|
| `monitor.py` — Bayesian online change-point detection (Adams & MacKay) over a Dirichlet-multinomial observation model | **Implemented, tested** (3 synthetic-data tests: detects an injected change, stays stable with none, streaming == batch; 1 regression test for a zero-prior-category NaN found while wiring observe.py) |
| `observe.py` — possession → observation vector | **Implemented, tested.** Flattens possessions into pitchgraph's TACTICAL grid (start zone, outcome) vocabulary — the same layout `pitchgraph.chains.counts.outcome_table` uses — so a team's already-fitted v3 shrunk chain (`chains.team.Chain.P`) plugs directly in as the Dirichlet prior with no leakage |
| `nulls.py` — permutation null for threshold calibration | Implemented, tested on synthetic data; still needs real possession sequences to calibrate for real |
| `describe.py` — before/after possession counts → plain-English sentences | **Implemented, tested.** Lane-share shifts, shots-per-possession, biggest zone-to-zone edge shift. Threat is approximated by shots/possession, not true xG — see the module's scope note |
| `attribute.py` — candidate preceding events | **Partially implemented, tested.** `candidate_events()` pulls subs/cards/goals/half-time/tactical-shift markers for both teams from the lake; `rank_candidate_causes()` currently ranks by temporal proximity only — a placeholder until `01_validate.ipynb`'s learned likelihood ratios exist |
| `export.py` | Still a stub — needs `attribute.py`'s real ranking and a decision on the match-JSON schema the (not-yet-built) Library timeline page will consume |
| `notebooks/01_validate.ipynb` | Not started |

**Known data gap found along the way:** pitchgraph's ingest drops StatsBomb's `tactics` payload entirely (excluded from both the event's own columns and the `extra` JSON catch-all, via `ingest.CORE_KEYS`). A "Tactical Shift" candidate can currently say *that* one happened, not what the formation changed to or from — needed for distinguishing the ~30% that are real changes per the design doc. Fixing this needs an `ingest.py` change and a re-ingest.

## Scope note on the detector as implemented

The design doc models each zone row as its own conditional multinomial (a full team chain). What's implemented and tested here is a flattened version: one Dirichlet-multinomial over all (zone, outcome) categories together (960 categories: 30 zones × 32 outcomes). This is an honest simplification, not the final per-row conditional model the design doc describes — that's the natural refinement once this flattened version has been validated against real matches.

## Design

Full design, ground truth, and validation plan: see the PitchGraph programme plan (Part B).
