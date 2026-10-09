# dynamic_shift — live match monitor

*When did this game change, what changed, and what probably caused it?*

Second project in the PitchGraph programme (after [pitchgraph](https://github.com/akabra13/football_graph_predictor) v3, before RoleFit). Depends on pitchgraph's v3 team flow graphs, pinned to commit `3b773a1`.

**Honest scope:** open data is historical, so "live" means the detector is *online* — each decision at time t uses only events up to t. A replay of a past match is then faithful to what a live feed would have seen, and `streaming == batch` is enforced by a test.

## Status (2026-10-08)

The pitchgraph Drive lake and Library are built (all 80 competition-seasons, 13.9M events; 23 competition-seasons in the Library with fitted team chains), and the lake now carries StatsBomb's `tactics` payload (formation + lineup), which the ground-truth classification below needs.

| Piece | State |
|---|---|
| `monitor.py` — Bayesian online change-point detection (Adams & MacKay) over a Dirichlet-multinomial observation model | **Implemented, tested** (3 synthetic-data tests: detects an injected change, stays stable with none, streaming == batch; 1 regression test for a zero-prior-category NaN found while wiring observe.py) |
| `observe.py` — possession → observation vector, plus possession → clock-time lookup | **Implemented, tested.** Flattens possessions into pitchgraph's TACTICAL grid (start zone, outcome) vocabulary — the same layout `pitchgraph.chains.counts.outcome_table` uses — so a team's already-fitted v3 shrunk chain (`chains.team.Chain.P`) plugs directly in as the Dirichlet prior with no leakage |
| `nulls.py` — permutation null for threshold calibration | Implemented, tested on synthetic data; still needs real possession sequences to calibrate for real |
| `describe.py` — before/after possession counts → plain-English sentences | **Implemented, tested.** Lane-share shifts, shots-per-possession, biggest zone-to-zone edge shift. Threat is approximated by shots/possession, not true xG — see the module's scope note |
| `attribute.py` — candidate preceding events | **Partially implemented, tested.** `candidate_events()` pulls subs/cards/goals/half-time/tactical-shift markers for both teams from the lake; `rank_candidate_causes()` currently ranks by temporal proximity only — a placeholder until real likelihood ratios are learned from the validation run below |
| `ground_truth.py` — positive/negative controls | **Implemented, tested.** Classifies Tactical Shift events as real formation changes vs same-formation (using the `tactics` payload), finds red cards, checks for a nearby substitution. "Like-for-like subs" (also a negative control in the design doc) isn't done yet — needs lineup position data |
| `validate.py` — leave-one-match-out detection check | **Implemented, tested** on a hand-constructed scenario (stable training matches, a held-out match with an injected shift) — not yet run against real lake data (see below) |
| `export.py` | Still a stub — needs a decision on the match-JSON schema the (not-yet-built) Library timeline page will consume |
| Real validation run across the lake | **Not run yet.** Needs this repo's own Drive access (`GDRIVE_OAUTH_TOKEN`/`GDRIVE_FOLDER_ID` secrets, same setup as pitchgraph's `docs/github_actions_drive_setup.md`) and a GitHub Actions workflow mirroring pitchgraph's `ingest.yml`/`library.yml` pattern |

## Scope note on the detector as implemented

The design doc models each zone row as its own conditional multinomial (a full team chain). What's implemented and tested here is a flattened version: one Dirichlet-multinomial over all (zone, outcome) categories together (960 categories: 30 zones × 32 outcomes). This is an honest simplification, not the final per-row conditional model the design doc describes — that's the natural refinement once this flattened version has been validated against real matches.

## Design

Full design, ground truth, and validation plan: see the PitchGraph programme plan (Part B).
