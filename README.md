# TurningPoint — live match monitor

*When did this game change, what changed, and what probably caused it?*

Second project in the PitchGraph programme (after [pitchgraph](https://github.com/akabra13/football_graph_predictor) v3, before RoleFit). Depends on pitchgraph's v3 team flow graphs, pinned to commit `3b773a1`.

**Honest scope:** open data is historical, so "live" means the detector is *online* — each decision at time t uses only events up to t. A replay of a past match is then faithful to what a live feed would have seen, and `streaming == batch` is enforced by a test.

## Status (2026-10-07)

| Piece | State |
|---|---|
| `monitor.py` — Bayesian online change-point detection (Adams & MacKay) over a Dirichlet-multinomial observation model | **Implemented, tested on synthetic data** (3 tests green: detects an injected change, stays stable with none, streaming == batch) |
| `observe.py` — possession → observation vector | Stub. Needs the zone/outcome vocabulary + possession data from the pitchgraph Drive lake |
| `nulls.py` — permutation null for threshold calibration | Implemented, tested on synthetic data; needs real possession sequences to calibrate for real |
| `describe.py`, `attribute.py`, `export.py` | Stubs with the intended interface; blocked on real possession sequences and the validation suite |
| `notebooks/01_validate.ipynb` | Not started |

**Blocked on:** the pitchgraph Drive lake. `notebooks/01_ingest.ipynb` and `02_library.ipynb` in the pitchgraph repo haven't been run yet (pending a Colab session). No soccer data is stored in this repo or on the laptop — see pitchgraph's data policy, which this repo inherits.

## Scope note on the detector as implemented

The design doc models each zone row as its own conditional multinomial (a full team chain). What's implemented and tested here is a flattened version: one Dirichlet-multinomial over all (zone, outcome) categories together. This is the honest simplification needed to have something real and tested before the actual zone vocabulary exists; `observe.py` is where the two meet, and upgrading to the per-row conditional model is a natural refinement once real possession sequences are available to validate against.

## Design

Full design, ground truth, and validation plan: see the PitchGraph programme plan (Part B).
