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

## Model-free signal check (2026-10-09, full lake)

Before tuning the detector further, `scripts/run_signal_check.py` asked a simpler question with no change-point model: does the 15 possessions after a real tactical change differ from the 15 before more than at an arbitrary moment? Coarse vocabulary (thirds x lanes = 15 destination cells), primary statistic fixed in advance (total-variation distance of the destination distribution). 80 competition-seasons, windows kept inside one half.

| Class | n | mean TV | AUC vs random |
|---|---|---|---|
| random moments | 12,995 | 0.3153 | - |
| same-formation shift (negative control) | 2,147 | 0.3152 | 0.499 |
| real formation change | 866 | 0.3236 | 0.527 |
| red card | 60 | 0.3311 | 0.550 |

- The negative control sits exactly on the random baseline, so the check itself is clean.
- Real formation changes are separated from random only by ~0.008 TV (~2.6% relative), AUC 0.53. A faint real effect, far too small to detect on any single event. Shot rate, turnover rate and actions per possession show nothing (AUC 0.50-0.52).
- **Conclusion:** in-possession ball movement carries almost no per-event signal of a formation change, at either 960 or 15 categories. The vocabulary was not the main problem; the observable is. Next: out-of-possession shape from StatsBomb 360 freeze-frames. Tuning the BOCPD vocabulary further is not justified.

## 360 shape check (2026-10-09)

Same question, now with team shape from StatsBomb 360 freeze-frames (`scripts/run_shape_check.py`, `shape.py`): visible outfield players of the team, mirrored to its attack direction, averaged over the 5 minutes before vs after each event inside one half, split by phase. 12 competition-seasons with 360 data (Euros 2020/2024, World Cups, Women's Euro 2022/2025, La Liga 2020/21, Ligue 1, Bundesliga 2023/24, MLS, AFCON), 834 team-matches. Primary statistic fixed in advance: |change| in out-of-possession depth relative to the ball.

| Class (primary stat) | n | mean abs change | AUC vs random |
|---|---|---|---|
| random moments | 3,450 | 4.26 | - |
| same-formation shift | 662 | 4.55 | 0.529 |
| real formation change | 287 | 4.23 | 0.505 |
| red card | 18 | 4.90 | 0.570 |

- Across all 8 shape statistics (depth, depth relative to ball, width, compactness; in and out of possession) real formation changes sit at AUC 0.49-0.55, indistinguishable from same-formation shifts and random moments.
- The noise floor is large: a random 5-minute split already moves mean block depth by ~10.6 pitch units, because freeze-frames only contain the players the camera sees.
- **Conclusion:** neither in-possession ball movement nor crude 360 shape separates a labelled formation change from an arbitrary moment. Either StatsBomb "formation change" labels rarely correspond to a change that is visible at this resolution, or the crude features cannot see it. A better test would need player identities in the frames (not provided) to compare role-specific positions.

## Label check (2026-10-12)

`scripts/run_label_check.py` compared the lineup before and after every StatsBomb Tactical Shift (10,860 shifts, 3,290 labelled real formation changes). The "labels are cosmetic" explanation is rejected: only 0.3% of labelled changes move nobody, the median labelled change moves 4 continuing players to new positions (same-formation shifts: mean 1.6). Caveat: 85% of labelled changes also involve a substitution or card; only 15% (~500) are moved>=3 with nobody joining/leaving.

## Pre-registered: graph / topology feature test (written before running)

Question: do richer structural features separate real tactical changes from random moments, where the earlier coarse ones did not (best AUC 0.53)?

- **Events:** `real_substantive` = labelled formation change with moved>=3; `real_clean` = moved>=3 and nobody joined/left; `same_formation` (negative control); `random` (5 per team-match) baseline.
- **Windows:** 25 team possessions before vs after for pass geometry and passing network; 5 minutes before vs after for freeze-frame topology (360 seasons only). Windows must be complete and inside one half.
- **Features:** pass geometry (length mean/std, forward component, lateral component, direction concentration, long-pass share); passing network (edge entropy, algebraic connectivity, weighted clustering, centralisation); freeze-frame persistent homology (H0 total and max = minimum-spanning-tree gaps, H1 total persistence and loop count), each out of and in possession.
- **Three primary statistics**, one per family: Euclidean norm of the per-feature signed change, each feature divided by its standard deviation among random moments. Per-feature results are secondary and descriptive only.
- **Pass bar:** a family counts as signal only if its AUC vs random for `real_substantive` is >= 0.60 AND the 98.3% bootstrap-by-team-match CI (Bonferroni for three primaries) excludes 0.5. `same_formation` must stay near 0.5 or the result is distrusted. Anything else is reported as no signal. No changes to features, windows or thresholds after seeing results.

### Result (2026-10-12)

**Amendment, disclosed:** the 25-possession window was infeasible. A team has only ~25-35 possessions per half, so almost no event had 25 on each side inside one half: the first graph run kept 37 real_substantive events (random 809), too few to conclude anything. I re-ran the graph/network families at 15 possessions, the window the earlier checks used. This was chosen after seeing sample sizes and after seeing the n=37 AUCs (0.58 geometry, 0.54 network, CIs spanning 0.5), so treat it as one extra forking path. Nothing else changed. (Also, only 23 competition-seasons are eligible for these runs, not 80: tournaments have too few matches per team for pitchgraph's league preparation.)

**Freeze-frame topology** (5 min windows, 12 360-seasons, run exactly as registered): real_substantive n=164, AUC 0.523 (98.3% CI 0.466-0.570); real_clean n=31, AUC 0.499 (CI 0.35-0.63); same_formation 0.519. Fails the bar. Every per-feature AUC is within 0.46-0.59.

Composites need all 8 phase-split features, so 301 labelled events shrank to 164 usable ones.

**Pass geometry and passing network** (15 possessions, 23 eligible seasons):

| Family | real_substantive (n=729) | real_clean (n=83) | same_formation (n=2147) |
|---|---|---|---|
| pass geometry | AUC 0.546 (CI 0.522-0.571) | 0.592 (0.515-0.670) | 0.505 |
| passing network | AUC 0.535 (CI 0.511-0.561) | 0.568 (0.494-0.632) | 0.510 |
| freeze-frame topology | AUC 0.523 (0.466-0.570) | 0.499 (n=31) | 0.519 |

**Verdict: none of the three families meets the pre-registered bar (AUC >= 0.60).** Geometry and network show a faint but real effect for substantive changes (CIs exclude 0.5; the negative control sits at 0.50-0.51), consistent with the earlier ball-movement result (0.53). It is a population-level tilt of a few AUC points, not a per-event detector. The cleanest subset trends higher (0.59 geometry) but n=83 and the CI includes 0.5-0.6, so that is a lead, not a finding.

**Conclusion for dynamic_shift:** across ball movement, 360 shape, pass geometry, passing networks and persistent homology, a labelled tactical change is only faintly distinguishable from an arbitrary moment. Event-level "when did the game change" detection from open StatsBomb data is not supported. What remains defensible is descriptive monitoring (what changed over a window, by team) without a claim of detecting tactical changes.

## Pre-registered v2: identity-aware tests (written 2026-11-09, before any run)

**Why try again.** Every family so far was identity-blind: zone distributions, pass-length statistics, network summary numbers, anonymous freeze-frames. A formation change is *specific players taking up new positions* (the label check found the median labelled change moves four continuing players), and events carry player ids and locations. That information has never been used. Earlier windows were also short (15 possessions, one half). This is a new observable, not a re-tune of an old one.

**Lockbox.** Competition-seasons whose key's MD5 hash is divisible by 4 (`split.is_lockbox`) are never touched while developing. They are scored once, only if a test passes on the development seasons. Everything below is fixed now.

**Test A, player position change** (`positions.py`, `scripts/run_position_check.py`).
- Events: the 10 outfield players' located events (all types) for the team. Time runs on one continuous clock across the two halves, so a window may straddle half-time; extra time is ignored.
- Windows: 600 s before vs 600 s after a moment, both complete and inside the match. A player counts if they have at least 5 touches in each window; at least 6 such players are needed.
- Classes: the existing `classify_shifts` (`real_substantive`, `real_clean`) and the negative control `same_formation`, against 5 random moments per team-match as the baseline.
- Two primary statistics, each a family: `disp_rel` (mean distance a continuing player's centroid moved, after subtracting the team's average shift) and `order_flip` (share of player pairs whose left/right or ahead/behind order reversed). Reported with the existing `report.family_report`.
- **Pass bar (unchanged):** AUC vs random for `real_substantive` at least 0.60 and the 98.3% bootstrap-by-team-match CI above 0.5 (Bonferroni over three primaries). `same_formation` must stay near 0.5.

**Test B, can a formation be read back from where players touch the ball** (`formation.py`, `scripts/run_formation_check.py`).
- Examples: every segment between two Starting XI or Tactical Shift events of at least 600 s, labelled by that event's formation. Features use only the ten lineup outfield players' touch centroids (at least 5 touches each), sorted by depth: their x values and widths. Position names are not used.
- Targets: back line size (3, 4 or 5) and number of forwards (1, 2 or 3), from the formation's first and last digits.
- Model: gradient boosting (200 iterations, learning rate 0.1, balanced class weights), 5-fold cross-validation grouped by match on development seasons.
- **Pass bar:** balanced accuracy at least 0.60 (chance 0.33) and the lower end of the 95% bootstrap-by-match CI above 0.40, for `back_line` (primary). `forwards` is secondary.
- If B passes, a follow-up (B2) applies the trained classifier to before/after windows and asks whether a change of predicted shape separates labelled changes from random moments. B2 is specified only after B's result and is not part of this pre-registration.

**Decision rule.** If A or B passes on the development seasons, rerun once on the lockbox. If it passes there too, rebuild the product on that observable and build `export.py` as a match timeline. If both fail on development, stop for good and publish the combined negative result. Either way it is written up.

### v2 development result (2026-11-09, run exactly as registered)

**Test A (position change).** On the numeric bar it passes, but the registered control invalidates it.

| Class | disp_rel AUC (98.3% CI) | order_flip AUC (98.3% CI) |
|---|---|---|
| real_substantive (n=1,728) | 0.625 (0.608-0.641) | 0.613 (0.596-0.629) |
| real_clean (n=318) | 0.621 (0.582-0.658) | 0.631 (0.595-0.670) |
| **same_formation, negative control (n=4,887)** | **0.636 (0.627-0.647)** | **0.624 (0.614-0.635)** |

The registered rule says `same_formation` must stay near 0.5 or the result is distrusted. It sits at 0.62-0.64, at or above the real changes. So what separates these moments from random ones is not a formation change: players move about more around *any* logged tactical event (substitutions, injuries, restarts) than at an arbitrary moment. Verdict: **no evidence that position change identifies a formation change.** Lockbox not run.

**Test B (formation recognition).** Fails. Back line: balanced accuracy 0.506 (95% CI 0.497-0.514) against the 0.60 bar, chance 0.33; n = 11,768 segments, but the classes are 4 (10,070), 3 (1,692) and 5 (6), so the 3-class balanced figure is close to a 2-class problem. Forwards: 0.486 (0.477-0.495). Where players touch the ball does not recover the formation label, even though the label is a clear fact about the lineup. Lockbox not run.

## Scope note on the detector as implemented

The design doc models each zone row as its own conditional multinomial (a full team chain). What's implemented and tested here is a flattened version: one Dirichlet-multinomial over all (zone, outcome) categories together. This is an honest simplification, not the final per-row conditional model the design doc describes.

## Design

Full design, ground truth, and validation plan: see the PitchGraph programme plan (Part B).
