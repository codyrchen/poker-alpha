# Solver validation (Phases 32-37)

Is the MCCFR implementation correct, and are the suspicious preflop outputs
of the locked strategy (`HoldemSolverConfig:v1`, compact encoder, imperfect
recall) caused by bugs or by abstraction and training? Every number here is
from a committed result file:

* `results/validation/preflop_audit_v1.json` — 169-class audit (32A),
  heatmaps in `results/figures/phase32/`;
* `results/validation/solver_quality_v1.json` — rules, geometry, utility,
  MCCFR trace, averaging windows, action EVs, recall, jams (32B/32C).

## Phase 32 — solver trustworthiness

### 32A Full 169-class preflop audit

10 situations x 169 classes x 9 checkpoints (seeds 0/1/2 at 10k / 100k /
300k): key, visits, average and current strategy, regret and strategy sums,
movement and seed disagreement (`experiments/phase32_preflop_audit.py`).

| situation | median visits @300k | seed disagreement (mean L1, max 2) | 100k->300k movement (L1) | mean jam freq. | classes jamming > 50% |
| --- | --- | --- | --- | --- | --- |
| BTN first action | 1,363 | 0.90 | 0.40 | 5.1% | 0 |
| BB vs limp | 1,364 | 0.86 | 0.42 | 9.9% | 4 |
| BB vs open 33% (to 1.66 BB) | 1,364 | 0.95 | 0.44 | 16.8% | 14 |
| BB vs open 75% (to 2.5 BB) | 1,364 | 0.90 | 0.43 | 13.3% | 10 |
| BB vs open 150% (to 4 BB) | 1,364 | 0.84 | 0.46 | 13.2% | 4 |
| BB vs all-in | 1,364 | 0.38 | 0.21 | — | — |
| BTN limp vs raise 33/75/150% | 119 | 1.13-1.17 | 0.29-0.33 | 18-20% | 4-10 |
| BTN open 75% vs 3-bet 75% | 688 | 1.07 | 0.38 | 11.0% | 6 |

**Main finding: the preflop strategy is dominated by noise.** Every
first-action infoset has ~1,360 visits at 300k, yet independently seeded
runs disagree by L1 ~0.9 on the same class, and each run still moves ~0.4
between 100k and 300k. 1,107 of 1,690 class-situation states exceed L1 0.8
seed disagreement. Clear-cut hands are stable (72o first in: fold 93%, seed
L1 0.13; facing a jam, L1 0.38); close decisions are not.

### 32B AA limping (BTN first, 61% limp averaged over seeds)

| hypothesis | test | result |
| --- | --- | --- |
| rules bug | HU order, blinds, pot after limp-check (2.0), all-in and call amounts (100), 3-bet effective stack | **ruled out** (all pass; `tests/test_solver_trust.py`) |
| utility bug | SB fold -0.5, BB fold to open +1, raise-call-checkdown +2.5, all-in-call +/-100, split 0 | **ruled out** |
| MCCFR update bug | instrumented updates at the AA infoset: node value = strategy . child values, regret delta sums to 0 under the strategy, strategy sum untouched on traverser visits, fold child = -0.5; first-action infosets get exactly one average-strategy visit per iteration | **ruled out** in Hold'em; exact-game convergence in Phase 33 |
| action abstraction | concrete sizes: BTN open 33% = **to 1.66 BB** (below the 2 BB legal minimum), raise over a limp 33% = to 1.66 BB; 81% of preflop `b33` offers and ~30% of postflop `b33` bets/raises are **below NLHE minimums** | **contributes** (sizes are not legal NLHE; the game is internally consistent but mis-shaped preflop) |
| averaging effect | full average vs windows: seed 0 limp 62% (full), 49% (100k-300k window), current 55%; the first 10k iterations hold only ~3% of AA's visits | **not the cause** (recent windows and current strategies also limp 40-82%) |
| imperfect recall | the AA first-action key is the exact class (no merge). After a limp the flop keys AA reaches merge it with other holdings and lines (e.g. one top-pair+ limped-pot key holds 16 different preflop classes; others mix 4 preflop lines) | **contributes** to the continuation values, not to the preflop key itself |
| undertraining / sampling noise | EVs of forced first actions under the trained 300k profile (10,000 paired deals): raise 75% beats limp by **+1.12 BB (SE 0.46)** (seed 0) and **+0.96 (0.46)** (seed 1); jam is -3.9 / -2.8 BB; individual sampled child values range over +/-100 BB | **primary cause**: limping is not a best response to the profile, but the ~1 BB gap is small against the per-sample variance of ~1,300 regret updates |
| equilibrium feature | — | **unresolved**: no exact solution of this abstract game exists to compare |

**Diagnosis: UNDERTRAINING (sampling noise at 100 BB) + ACTION ABSTRACTION
+ IMPERFECT-RECALL EFFECT. Not a rule, utility, MCCFR or averaging bug.**

### 32C Preflop all-ins

* Highest where the open is tiny: BB vs the 1.66 BB "33%" open jams with 14
  classes > 50% (16.8% mean).
* EVs under the profile (8,000 paired deals): BB 99 vs open33 — every raise
  incl. jam beats calling by ~4-6 BB, jam within noise of raising; AJo vs
  open75 — jam best (+5.3 vs call, SE 0.58); A5o vs open33 — every action
  within ~1 SE (jams 77% of the time: noise); 72o vs open33 — jam -11 BB, and
  72o rarely jams. Jams are roughly EV-maximizing against a BTN that folds
  to a jam with ~66% of classes.
* **Imperfect-recall collision found:** the BTN's information set facing a
  jam is the same after a 33%, 75% or 150% open (`b33a` / `b75a` / `b150a`
  map to one key: the betting context keeps "facing all-in", not the size
  of the BTN's own open).
* Utility of all-in lines is exact (32B). Diagnosis: **pot-relative opening
  sizes + near-indifference noise + response collision; not a utility bug.**

## Phase 33 — exact Hold'em-shaped games

`poker_alpha/games/reduced_holdem.py`, `experiments/phase33_reduced_games.py`
-> `results/validation/reduced_holdem_v1.json`. Exploitability is exact for
the defined games (mean of both players' best-response gains, in BB / chips).

**Reduced preflop game** (deck A K Q J T x 2 suits, 45 hands; 0.5 / 1
blinds; 10 BB; fold / limp / raise to 2.5 / all-in; showdown by a seeded
equity table; 360 infosets). Reference CFR+ 2,000 iterations:
exploitability 3e-5, game value -0.0453 BB for the button.

| iterations | CFR | CFR+ | MCCFR seed 0 / 1 / 2 | MCCFR weighted L1 to reference |
| --- | --- | --- | --- | --- |
| 100 | 0.0077 | 0.0035 | 0.579 / 0.559 / 0.597 | 1.14 |
| 1,000 | 0.0017 | 0.0001 | 0.259 / 0.279 / 0.281 | 0.88 |
| 10,000 | — | — | 0.066 / 0.063 / 0.064 | 0.57 |
| 100,000 | — | — | 0.016 / 0.014 / 0.013 | 0.31 |
| 300,000 | — | — | 0.0070 / 0.0071 / 0.0079 | 0.24 |

MCCFR exploitability falls roughly as 1/sqrt(T) and the three seeds agree;
EV error is below 0.002 BB from 10k iterations. The weighted L1 to the
reference stays sizable (0.24 at 300k) because many actions are nearly
indifferent: a strategy can be close to unexploitable while its
frequencies differ from the reference's.

**Six fixed-board 52-card river subgames** (pot 10, 20 behind, bets 50% /
100% + all-in, cap 2; 24-combo ranges from strength-percentile bands; 336
infosets each): CFR+ references at 1,500 iterations have exploitability
<= 0.001. MCCFR at 10k / 30k / 100k iterations: 0.21-0.52 / 0.10-0.24 /
0.044-0.12 (all boards, all seeds), weighted L1 0.18-0.40 at 100k.

**Conclusion: external-sampling MCCFR as implemented converges to the exact
solution on Hold'em-shaped games.** No solver bug.

Rollout recommendation vs the exact equilibrium (OOP hero, six hands per
board, opponent range = the subgame range, 1,000 rollouts): agrees with the
equilibrium's majority action in **20 / 36** spots; all 16 disagreements are
the rollout betting (15 all-in) where the equilibrium checks — the default
behaviour model folds too much to large bets. Rollout EVs are correct for
their model (below); the model is the weakness.

Compact-strategy lookups are not compatible with these subgames (different
pot / stack / menu); no lookup was attempted.

## Phase 34 — abstraction error

`experiments/phase34_abstraction_error.py` -> `results/validation/abstraction_error_v1.json`.
Identical tree for every encoder: the river decision of the real
`HoldemGame` (line b75c / cc / cc: pot 5 BB, 97.5 BB behind, 33 / 75 / 150% +
all-in, cap 2), 14-combo ranges. Each encoder's abstract game is solved by
CFR+ (400 iterations) and the strategy is evaluated in the raw game.

| board | encoder | keys | exploitability in raw game (BB) | EV error | weighted L1 vs raw |
| --- | --- | --- | --- | --- | --- |
| dry K72Q4 | raw | 476 | 0.006 | — | — |
| | bucket (perfect recall) | 476 | 0.006 | 0 | 0 |
| | transition (perfect recall) | 476 | 0.006 | 0 | 0 |
| | **compact (v1)** | **99** | **9.24** | 0.44 | 0.19 |
| four-flush Q952K | raw | 476 | 0.019 | — | — |
| | bucket / transition | 476 | 0.019 | 0 | 0 |
| | **compact (v1)** | **63** | **2.24** | 0.32 | 0.51 |

On these small ranges the perfect-recall encoders merge nothing; the
compact encoder's river rung (0..7 made-hand ladder) merges hands that must
play differently, and the merged strategy is exploitable by a large margin
at all-in-heavy nodes even where average L1 is small.

**Collision attribution** (34B): the canonical-matrix flop keys with the
highest seed disagreement (L1 1.4-1.9) are mostly coherent: within-key
equity ranges 0-0.25, one or two adjacent made-hand classes, severity
0-0.32. Their instability is training noise, not card collisions; the
measured large error is on the river.

## Phase 35 — remediation

Evidence-backed changes, both opt-in (v1 is untouched and bit-identical):

1. **Legal NLHE sizing** (`enforce_min_raise`, preflop raise-to multiples
   2 / 2.5 / 3.5x): v1 offers sizes below the NLHE minimum (81% of preflop
   33% raises, ~30% of postflop 33% bets/raises) and the decision engine
   could recommend them. Additionally, the solver lookup now drops any
   sub-minimum size from every strategy (mass reported).
2. **Exact river percentile buckets** in the compact encoder
   (`river_percentile_buckets=20`): exploitability in the same exact river
   subgames

   | board | compact v1 | 10 buckets | **20 buckets** |
   | --- | --- | --- | --- |
   | dry K72Q4 | 9.24 | 8.02 | **0.57** |
   | four-flush | 2.24 | 0.49 | **0.28** |

   (`results/validation/abstraction_error_river_pct.json`). Cost: 4.4x
   slower training (each new river board needs 1,081 hand evaluations).

Rejected, with reasons: changing the averaging scheme (Phase 32 windows show
recent and current strategies limp AA as much as the full average, and MCCFR
already converges on the exact games); a separate preflop encoder (the
preflop key is already the exact 169 class); extra betting-context detail
for the facing-jam collision (it changes the required call equity by ~1
percentage point); premium-hand de-abstraction (no preflop merge exists).

`V2_CONFIG` = both changes; retraining is reported in Phase 36.

## Phase 37 — solver confidence gate

`poker_alpha/decision/solver_gate.py`. Signals per abstract key (from the
confidence table built from the training runs): visits, movement between
checkpoints, seed disagreement, collision dispersion, audit-flagged keys.
Output `SOLVER_ACCEPT` / `SOLVER_LOW_CONFIDENCE` / `SOLVER_REJECT` with
machine-readable reasons (`LOW_VISIT_COUNT`, `HIGH_SEED_DISAGREEMENT`,
`UNSTABLE_ACROSS_CHECKPOINTS`, `HIGH_COLLISION_DISPERSION`,
`CONFIG_MISMATCH`, `INCOMPATIBLE_CHECKPOINT`, `UNSEEN_STATE`,
`OUTSIDE_ABSTRACTION`, `KNOWN_PATHOLOGICAL_BUCKET`, `NO_STABILITY_DATA`).
Rejected lookups fall back to rollout, then heuristic.

**Calibration (37A)** — `experiments/phase37_calibration.py` ->
`results/validation/solver_gate_calibration.json`: 3 MCCFR seeds on the
reduced preflop game and two river subgames (3k and 30k iterations), 1,888
infosets with their true L1 error vs the exact solution. Spearman
correlation with true error: seed disagreement **0.61**, movement 0.33,
visits -0.06.

| seed disagreement (L1) | < 0.05 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-0.8 | 0.8-1.2 |
| --- | --- | --- | --- | --- | --- | --- |
| median true error | 0.014 | 0.17 | 0.26 | 0.40 | 0.56 | 0.84 |

Thresholds (lowest bin edge from which every bin's median true error
exceeds 0.5 / 0.25): reject at seed disagreement >= 0.5 or movement >= 0.5;
low confidence at >= 0.2 / >= 0.1. Visits did not predict error, so the
20-visit rule is a documented sanity floor, not a calibrated threshold. The
collision threshold (>= 3 ladder rungs mixed) is a heuristic.

Applied to the v1 300k strategy: 72% of keys rejected; weighted by visits,
37.7% of decisions rejected, 26.7% low confidence, 35.6% accepted. Reports
carry `details["solver"]` (used, confidence, visits, seed disagreement,
movement, collision, reasons) and the UI shows them.

## Phase 39 — strategic references

**Clairvoyant river toy game** (analytically solved; `experiments/phase39_reference_checks.py`
-> `results/validation/strategy_reference_checks.json`): OOP holds the nuts
or air (1 : 2), IP a bluff-catcher, one bet size s.

| s (pot) | value bets (theory 1) | bluff share of bets (theory s/(1+2s)) | IP calls (theory 1/(1+s)) |
| --- | --- | --- | --- |
| 0.5 | 1.000 | 0.2502 (0.25) | 0.6668 (0.667) |
| 1.0 | 1.000 | 0.3332 (0.333) | 0.5001 (0.5) |
| 2.0 | 1.000 | 0.3996 (0.4) | 0.3341 (0.333) |

Polarization grows with size as theory says. Broad principles on the
trained strategy (premiums aggressive, trash folds, nuts never fold, pot
odds, bigger bets get more folds) are the Phase 29 sanity checks (10 / 10
on every seed). No proprietary solver output was used.

## Rollout validation

`experiments/rollout_validation.py` -> `results/validation/rollout_validation_v1.json`
(5 seeds x {100, 400, 1,000, 5,000} rollouts, pinned response model):

| case | exact EV | mean abs error at 100 / 400 / 1,000 / 5,000 |
| --- | --- | --- |
| call vs all-in (pot odds) | 0.0 | 1.70 / 0.78 / 0.34 / 0.20 (SE 2.44 -> 0.35) |
| always-fold: bet wins the pot | 12.0 | 0 / 0 / 0 / 0 |
| value bet vs always-call | 3.0 | 1.02 / 0.47 / 0.20 / 0.12 |
| bluff, opponent folds 30% | -2.7 | 1.09 / 0.28 / 0.33 / 0.17 |
| bluff, opponent folds 60% | 3.6 | 1.13 / 0.24 / 0.20 / 0.09 |

Every run is within 3 SE of the closed form; error shrinks ~1/sqrt(n).
The weakness is the response model, not the estimator (Phase 33 above).

## Range / model sensitivity

`experiments/range_sensitivity.py` -> `results/validation/range_sensitivity_v1.json`.
Changing only the assumed opponent model changes the recommendation in two
of three HU spots: top pair facing a flop bet -> raise (regular), call
(nit), all-in (calling station, maniac); best-action EV 4.6-30.1 BB. River
bluff-catcher facing a pot bet -> fold (regular, nit, station), call
(maniac), raise (any two cards). A draw first to act is "all-in" under every
model (the known check-down bias of rollouts at high SPR). Inferred ranges
are model assumptions, and reports must be read that way.

## Phase 36 — training with the remediated config

`V2_CONFIG` (`HoldemSolverConfig:v2:733e52f1d1014e2e7973`), seeds 0 / 1 / 2,
checkpoints 1k / 5k / 10k / 30k / 100k
(`results/validation/holdem_training_v2.json`, `preflop_audit_v2.json`).
Runs were restarted at 5k with an exact, 3x faster river ranking (resume is
bit-exact); ~9 iterations/s per process, ~3 h per seed to 100k.

| iterations (seed 0) | infosets | new / iter | >= 5 visits | >= 20 visits | top-2000 movement | matrix movement | seeds: top-2000 overlap / common L1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k | 53,395 | 53.4 | 22% | 3.5% | — | — | 70% / 0.65 |
| 10k | 101,875 | 2.7 | 53% | 27% | — | 0.35 | 80% / 0.45 |
| 30k | 119,760 | 0.89 | 66% | 43% | 0.28 | 0.48 | 82% / 0.36 |
| 100k | 133,044 | 0.19 | 76% | 58% | 0.25 | 0.50 | 85% / 0.30 |

**v1 vs v2 at matched 100k** (`solver_quality_v1.json` -> `v1_vs_v2`):

| | v1 | v2 |
| --- | --- | --- |
| infosets | 82,781 | 133,044 |
| >= 5 / >= 20 visits | 79% / 62% | 76% / 58% |
| seeds: matrix L1 (both >= 20 visits) | 0.84 | 0.85 |
| seeds: top-2000 common L1 | 0.25 | 0.30 |
| sanity checks (each seed) | 10 / 10 | 10 / 10 |
| preflop seed disagreement (BTN first) | 0.94 | 0.96 |
| BTN AA: limp / raise / jam | 69% / 27% / 4% | 48% / 51% / 0.2% |
| BTN KK: limp | 49% | 26% |
| BTN 72o: fold | 80% | 88% |
| preflop jam frequency (BB vs medium open) | 16.7% | 17.6% |
| gate at 100k, visit-weighted accept / low / reject | 22% / 20% / 59% | 19% / 16% / 65% |
| sizes below the NLHE minimum | yes (filtered at lookup) | none by construction |
| exact river abstraction error (Phase 34/35 boards) | 9.24 / 2.24 BB | 0.57 / 0.28 BB |

Cross-play inside v2 (200,000 duplicate deals): seeds +14.7 (+4.9..+24.5),
+1.9 (-8.1..+11.8), +1.1 (-8.7..+10.9) bb/100; 100k vs 10k +270
(+258..+282); vs 1k +454; vs uniform +513; vs calling station +515;
uniform control -10.0 (-23.9..+4.0). Not significantly different is not
identical, and none of this is exploitability.

**Conclusions.** v2 fixes what was measurably wrong (illegal sizes, river
collisions) and makes premium preflop play more aggressive, but it does not
fix the dominant preflop problem: noise at ~450 visits per preflop key
(seed L1 ~0.95 at 100k in both configs). **Release config: v2**; v1 remains
the locked, reproducible reference.

**Stop criterion (36B/36C).** Training still improves (100k beats 10k by
~270 bb/100; v1 300k beat 100k by ~84), and policies still move (L1 ~0.25
top-2000, ~0.5 matrix). A 1M-iteration run was **not** started: v2 costs
~3 h per 100k iterations per seed even after the speedup (~30 h per seed to
1M), and its main beneficiary would be the noise-dominated preflop, which
the gate already rejects (all 169 BTN first-action keys) and routes to
rollouts. Extending three seeds to 300k (~6.5 h) is the recommended next
step.

## Phase 37 applied to the release strategy

`results/strategy/holdem_v2_seed0_confidence.npz` (124,381 visited keys;
seeds 0-2 at 100k, movement vs 10k). With the calibrated thresholds: 86% of
keys rejected, 8% low confidence, 6% accepted; weighted by visits 65% / 16%
/ 19%. All 169 BTN first-action preflop keys are rejected
(`HIGH_SEED_DISAGREEMENT`, `UNSTABLE_ACROSS_CHECKPOINTS`), so preflop advice
comes from rollouts; the solver contributes mainly in stable postflop
spots.

## Phase 38 — real screen fixtures

`tests/fixtures/pokernow/raw/` and `annotations/` are empty: **REAL SCREEN
VALIDATION: BLOCKED.** No screenshots were fabricated or scraped. The
tooling is ready (annotation validation, per-field metrics, confidence
calibration, harness self-test on the synthetic image); temporal (multi-frame)
metrics remain the synthetic Phase 25 ones. `observer_real_v1.json` is not
produced because no real fixtures exist.

## Best response feasibility (Phase 66)

`results/validation/best_response_feasibility.json`.

* The v2 betting tree has **191,100 decision nodes per deal** (preflop 106,
  flop 3,438, turn 31,006, river 156,550) and 313,314 terminals.
* **Exact best response in real Hold'em: infeasible here.** Even with suit
  isomorphism the public tree has ~6.5e11 river nodes, each carrying
  1,326-hand range vectors (>= 1e15 operations).
* **"Abstract" best response: not meaningful.** v2 is an imperfect-recall
  abstraction; a best response inside it is not a well-defined game value
  (and NP-hard in general), so no number is reported.
* **What exists:** exact exploitability on fixed-runout flop/turn subgames
  (Phases 56-57) and 52-card river subgames (Phase 34).
* **Feasible next step (not implemented): local best response (LBR)**, a
  sampled *lower* bound on real-game exploitability. Estimated 1-5 s per hand
  in this code base, so 10,000 hands ≈ 3-14 CPU-hours per strategy
  (SE ≈ 0.1 BB/hand).

## Confidence gate v2 (Phase 64)

Thresholds are **unchanged** (no new exact-game evidence to move them;
weakening the gate was out of scope). Changes:

* Two spot-level coded reasons, applied at lookup time, that can only lower
  the status (ACCEPT -> LOW, REJECT stays REJECT):
  `OFF_TREE_TRANSLATION` (observed sizes were mapped onto the abstract tree;
  previously only a free-text warning) and `ILLEGAL_SIZE_MASS` (>= 20% of the
  solver's mass was on sizes below the NLHE minimum and was removed —
  heuristic threshold).
* Acceptance by street: `python experiments/gate_acceptance.py
  results/strategy/holdem_v2_seed0_confidence.npz --out
  results/validation/gate_acceptance_v2_100k.json`.

Release strategy (v2, 100k iterations, 3 seeds), visit-weighted share of
self-play decisions:

| street | keys | accept | low | reject |
| --- | --- | --- | --- | --- |
| preflop | 4,250 | 8.7% | 12.8% | 78.4% |
| flop | 17,932 | 17.9% | 15.2% | 66.9% |
| turn | 29,932 | 20.0% | 15.8% | 64.1% |
| river | 72,267 | 21.0% | 16.3% | 62.8% |
| all | 124,381 | 19.5% | 15.8% | 64.7% |

All 338 preflop first-action keys are rejected (seed disagreement /
movement): in practice the solver is not used for the first preflop
decision, and recommendations there come from rollouts. Phase 65 checks
whether longer training changes this.

## Longer v2 training: 200k / 300k (Phase 65)

`results/validation/holdem_training_v2_300k.json` (phase29 analysis over all
checkpoints: seeds 0-2 to 200k, seed 0 to 300k; 200,000 duplicate deals per
cross-play match) and `results/validation/holdem_training_v2_decision.json`.

* **Play still improves** inside the abstract game: 200k beats 100k by
  66-69 bb/100 (95% CI 57-78), 300k beats 200k by 20-37 bb/100 (CI 12-45).
  The three seeds at 200k are indistinguishable from each other (-3 to +7
  bb/100, every CI includes 0). Cross-play is not an exploitability bound.
* **Seeds still disagree** on individual spots: canonical-matrix mean L1
  0.87 (100k) -> 0.82 (200k) of a maximum 2. Example, BTN unopened AA:
  limps 86% (seed 0) vs 36% (seeds 1, 2). First-in jam frequency 5-15% by
  seed at 200k (4% for seed 0 at 300k). Sanity checks 10/10 per seed.
* **The gate does not open up:** with a 200k confidence table built the
  same way, 18.0% of visit-weighted decisions are accepted (19.5% at
  100k); 335 of 338 preflop first-action keys are still rejected.
* **Promoted (owner decision):** seed 0 @ 200k with its 200k confidence
  table is now the release strategy (same config signature;
  `results/strategy/holdem_v2_seed0_200k.npz`). The 100k pair is kept as the
  previous release. Still EXPERIMENTAL: not converged, not GTO. No 1M run.

## Native backend validation and the 300k/1M study (post-RC, 2026-10)

The C++ training backend (`docs/native_solver.md`) is validated against this
document's reference implementation by construction: on a shared random
tape the two backends are bitwise-identical through 1,000 Hold'em
iterations (`tests/test_native_mccfr_parity.py`), the evaluator matches on
all 2,598,960 five-card hands, and 20k+ random reachable states match on
mechanics and infoset keys. The exact reduced-game results above therefore
transfer to the native solver: it computes the same updates the Python
solver computed when it converged on the reduced preflop game and the six
river subgames. Fresh native seeds 0-2 were trained to 300k and 1M
(`results/validation/native_training_{300k,1m}.json`); findings and the
promotion decision are summarized in `docs/release_status.md`. The
preflop noise finding of Phase 32 persists at 1M (canonical seed L1 up to
1.6 preflop), while median seed disagreement falls 0.617 -> 0.523 and the
1M candidate wins abstract-game cross-play against every earlier milestone
with 95% CIs excluding zero.
