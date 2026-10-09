# dynamic_shift — live match monitor

*When did this game change, what changed, and what probably caused it?*

Second project in the PitchGraph programme (after [pitchgraph](https://github.com/akabra13/football_graph_predictor) v3, before RoleFit). Depends on pitchgraph's v3 team flow graphs, pinned to commit `3b773a1`.

**Honest scope:** open data is historical, so "live" means the detector is *online* — each decision at time t uses only events up to t. A replay of a past match is then faithful to what a live feed would have seen, and `streaming == batch` is enforced by a test.

## Status (2026-10-09)

The pitchgraph Drive lake and Library are built (all 80 competition-seasons, 13.9M events), and the lake carries StatsBomb's `tactics` payload (formation + lineup), which ground-truth classification needs. A full, label-free-calibrated validation pipeline now runs end-to-end against real lake data via GitHub Actions (`.github/workflows/validate.yml` → `scripts/run_validation.py`).

| Piece | State |
|---|---|
| `monitor.py` — Bayesian online change-point detection (Adams & MacKay) over a Dirichlet-multinomial observation model | **Implemented, tested.** Vectorized over run hypotheses and touched categories (~21ms/match vs the original ~440ms), and fixed a real semantic bug: a possession touching a category the prior assigns zero probability to now correctly rules that run-length hypothesis out, rather than being silently ignored |
| `observe.py` | **Implemented, tested.** `chain_prior_alpha` now has a floor pseudo-count so one rare unseen move doesn't trip the above impossibility on its own; `match_possession_counts` carries each possession's period, needed for half-preserving shuffles |
| `nulls.py` — permutation null for threshold calibration | **Implemented, tested**, and now actually used: shuffles within halves (never crossing half-time), scores once per permutation, sweeps thresholds cheaply over the already-computed trajectories |
| `describe.py` | Implemented, tested. Lane-share shifts, shots-per-possession, biggest zone-to-zone edge shift. Threat is approximated by shots/possession, not true xG — see the module's scope note |
| `attribute.py` | Partially implemented, tested. `candidate_events()` is real; `rank_candidate_causes()` is still a temporal-proximity placeholder |
| `ground_truth.py` | Implemented, tested. Carries possession numbers (not just clock minutes) so controls align correctly with the detector's possession-indexed output |
| `validate.py` | **Implemented, tested**, rewritten around `score_match` (run the detector once per team-match) + `alarms` (burn-in + refractory, not "every possession above threshold") + possession-number alignment |
| `export.py` | Still a stub |

## Real validation result (2026-10-09)

First full real-data run, FA WSL 2018/19 (key `37_4`), 214 team-matches, hyperparameters fixed *before* seeing any result (hazard=100, prior_strength=10, burn_in=10, refractory=5, change_point_window=5, within_possessions=15, 20 permutations/match):

- **The permutation null never got under the 0.25-false-alarm-per-match target** — even at the most conservative threshold tried (0.95), the null rate was 0.367. The threshold grid should extend closer to 1.0 to find where it actually crosses target; this wasn't done yet.
- At that threshold, detection rates for all four control classes are low and **statistically indistinguishable**: real formation changes 2.4% (90% CI 0–7.3%, n=37 matches), same-formation shifts 7.7% (CI 3.7–12.5%, n=96), red cards 0% (n=3, too few to read anything into), and — tellingly — **random time points with no claimed significance at all detect at 5.1%** (CI 3.3–7%, n=214), overlapping every other bucket.
- **Reading this plainly: at this observation granularity, the detector shows no demonstrated ability to tell a real tactical change from an arbitrary moment in the match.** This is a real result, not a pipeline bug — the code was checked at every step (unit tests on hand-traceable data, a timing benchmark, a smoke test on a synthetic lake) before trusting it on real data.
- Leading hypothesis: the 960-category (30 zones × 32 outcomes) vocabulary is too sparse. A possession has ~5 actions on average, so each one barely populates the space the Dirichlet-multinomial has to reason over — there may just not be enough evidence per possession for BOCPD to separate signal from noise at this resolution.
- **Not yet run:** the full eligible lake (only one competition-season so far), and an ablation to a coarser observation vocabulary (e.g. thirds × lanes, ~15 zones) to test the sparsity hypothesis directly.

Raw numbers: `validation_results.json`, uploaded as a workflow artifact on every `validate.yml` run.

## Scope note on the detector as implemented

The design doc models each zone row as its own conditional multinomial (a full team chain). What's implemented and tested here is a flattened version: one Dirichlet-multinomial over all (zone, outcome) categories together. This is an honest simplification, not the final per-row conditional model the design doc describes.

## Design

Full design, ground truth, and validation plan: see the PitchGraph programme plan (Part B).
