# Release status (release candidate, Phase 40)

Machine-readable: [`results/validation/release_candidate.json`](../results/validation/release_candidate.json).
Evidence: [solver_validation.md](solver_validation.md) (Phases 32-40) and
[validation.md](validation.md) (Phases 25-29). Every number traces to a
committed file in `results/validation/`.

**What PokerAlpha is not:** a solver for no-limit Hold'em, a GTO or Nash
strategy for real Hold'em, a profitable bot, or a validated PokerNow reader.
The heads-up strategy is an abstract MCCFR strategy on an **imperfect-recall**
abstraction with **no equilibrium guarantee**; exploitability of the full
abstraction is not computed. The observer is read-only and never acts.

## Component status

| status | component | evidence |
| --- | --- | --- |
| VALIDATED | Kuhn / Leduc research core | exact exploitability; pinned digests unchanged |
| VALIDATED | Hold'em rules engine (2-9 seats, side pots, all-in) | property tests; solver-game rules and terminal utilities checked (Phase 32) |
| VALIDATED | external-sampling MCCFR implementation | converges to exact solutions on a reduced preflop game (exploitability 0.007 BB at 300k, 3 seeds) and six 52-card river subgames; update identities traced in Hold'em |
| VALIDATED | solver tooling: config-bound checkpoints / artifacts, config inference, resume | signature tests; bit-identical resume and optimizations |
| VALIDATED | rollout estimator | converges to closed forms within 3 SE (100-5,000 samples) |
| VALIDATED | pipeline: solver -> rollout -> heuristic with a confidence gate and legal-size filter | tests over all four input sources; rejection reasons machine-readable |
| PARTIALLY VALIDATED | compact abstraction (v2) | river, FLOP and TURN error all measured on exact subgames (river cut 4-16x by pct20; flop median 3.7 bb / turn 5.1 bb exploitable at 5 bb pots) — docs/abstraction_error_summary.md |
| PARTIALLY VALIDATED | gate thresholds | calibrated on exact games (seed disagreement vs true error, Spearman 0.61), applied to the full abstraction by extrapolation |
| PARTIALLY VALIDATED | decision response models | rollouts disagree with exact river equilibria in 16 / 36 spots (they overbet); recommendations change with the assumed opponent model |
| PARTIALLY VALIDATED | screen observer on synthetic images | synthetic fixtures; annotation validation and metrics harness ready |
| EXPERIMENTAL | trained HU strategy (v2; release = NATIVE seed 0 @ 2M, SolverConfidence v2) | sanity green at 2M; seed disagreement still material but measured to be ~94% near-equivalent mixing preflop; restricted-LBR floor >= ~100 bb/100 (docs/solver_status_final.md) |
| EXPERIMENTAL | range and opponent modelling | beliefs under heuristic priors; no ground-truth accuracy |
| EXPERIMENTAL | real PokerNow recognition | heads-up preset aligned on 1 real annotated frame; all fields correct on it at 0.8x-3x, but the same frame was used for tuning, so accuracy is **not measured** (post-RC, `docs/observer.md`) |

## Release solver

| | |
| --- | --- |
| config | `HoldemSolverConfig:v2:733e52f1d1014e2e7973` (compact encoder + 20 river percentile buckets, legal NLHE sizing, 100 BB, raise cap 3, external-sampling MCCFR, uniform averaging) |
| artifact | **`results/strategy/holdem_v2_native_seed0_2m.npz`** (native backend, seed 0, 2,000,000 iterations, SHA-256 `f1860de7...`; promoted by the final-trust project) |
| confidence table | **`results/strategy/holdem_v2_native_seed0_2m_confidence.npz`** (SolverConfidence v2: seeds 0-2 at 2M, recent movement 1.5M->2M, SHA-bound; SHA-256 `ef5c3223...`) |
| use at decision time | gate (v2, after street caps): preflop 34% / river 53% visit-weighted accepted, flop & turn capped at LOW (72-77% low), rest rejected -> rollout / heuristic (results/validation/decision_pipeline_risk_summary.json) |
| status | **EXPERIMENTAL** — imperfect-recall abstraction, no equilibrium guarantee, not GTO; restricted-LBR lower bound >= ~100 bb/100 in the full game; training stopped at 2M by evidence (post_2m_training_decision.json) |
| previous release | `results/strategy/holdem_v2_seed0.npz` + `_confidence.npz` (seed 0, 100k) — archived in place (`solver_config.PREVIOUS_RELEASE_STRATEGIES`) so Phases 36-80 results stay reproducible |

## Performance (4-vCPU container, CPython 3.13, pure Python + NumPy)

Offline (historical, that container): v2 training ~9 it/s per process
(~3 h per seed to 100k); v1 ~30 it/s. **The optional native C++ backend
(`pip install ./cpp`, `docs/native_solver.md`) now trains the same release
config at ~837 it/s warm on an M3 Pro — a measured 38x over the Python
reference on that machine (22 it/s), with bitwise random-tape parity. Three
seeds reach 300k in ~7 minutes and 1M in ~28 minutes (parallel processes).** Online (median, `results/validation/latency_benchmark.json`):

| step | median | p95 |
| --- | --- | --- |
| solver lookup incl. gate | 0.016 ms | 0.025 ms |
| hand evaluation x2000 | 8.1 ms | 8.6 ms |
| HU equity, 2,000 sims | 37 ms | 38 ms |
| multiway equity (2 opp.), 2,000 sims | 115 ms | 118 ms |
| range update (1,326 combos) | 4.4 ms | 4.5 ms |
| rollout 100 / 400 / 1,000 | 34 / 86 / 192 ms | 35 / 88 / 193 ms |
| observer frame (synthetic) / fusion | 74 ms / 0.08 ms | 74 ms / 0.13 ms |
| full DecisionReport: solver path / heuristic / rollout 400 | 77 / 77 / 163 ms | 78 / 79 / 165 ms |

## Native-backend training study (post-RC, 2026-10)

Trained with the validated native backend, same config, fresh xoshiro seeds
0-2 (no resumable Python checkpoints survive, so these are new independent
seeds, Plan B): milestones to 300k and 1M, matching confidence tables at
both (seeds 0-2, movement vs 10k — the release construction), evidence in
`results/validation/native_training_{300k,1m}.json`.

| milestone | median seed disagreement | median visits | gate accept (visit-weighted) | cross-play vs release 200k |
| --- | --- | --- | --- | --- |
| 200k (release, python lineage) | 0.617 | 74 | 18.0% | — |
| 300k native | 0.595 | 106 | 18.1% | +27/-1/+23 bb/100 (CIs include 0) |
| 1M native | **0.523** | **327** | 16.2% | **+52..+56 bb/100 (CIs exclude 0)** |

1M also beats its own 300k ancestors by +55..+74 bb/100 (CIs exclude 0;
`results/validation/crossplay_1m_vs_300k.json`), and consecutive movement
per 100k iterations keeps falling (0.110 → 0.047 → ~0.017): the abstract
game is still genuinely improving at 1M. Strategic sanity at 1M: premiums
never fold to jams (AA/KK/QQ/AKs call ~100%, seed L1 < 0.01), AA 4-bets
94%, 72o folds 98% first-in.

**Best candidate: `results/strategy/candidates/native_1m/holdem_v2_native_seed0_1000k.npz`
with its matching 3-seed confidence table.** NOT auto-promoted: visit-weighted
gate acceptance falls to 16.2% because the movement signal compares against a
fixed 10k snapshot (a construction that penalizes longer training), and
canonical preflop seed disagreement remains severe (L1 up to 1.6) — two
Phase-63 no-promote triggers. Promotion is an owner decision; everything
needed (artifacts, table, gate report, cross-play) is committed. Cross-play
is abstract-game head-to-head, not exploitability; all candidates remain
EXPERIMENTAL.

## Readiness

| area | rating |
| --- | --- |
| Research | **READY WITH CAVEATS** — measured, reproducible, limits documented |
| Hand analysis | **READY WITH MODEL CAVEATS** — EVs are conditional on assumed ranges / response models |
| Private / play-money / test decision support | **READY WITH MANUAL STATE VERIFICATION** — read-only; verify recognized state; only where permitted |
| Real screen observation | **EXPERIMENTAL** — PokerNow heads-up only, 1 real frame (also the tuning frame); verify every reading |
| Trusted solver recommendations | **EXPERIMENTAL** — gated, abstract, not converged |

## Next steps that would change a rating

1. More real annotated PokerNow screenshots (not used for tuning; boards, other ranks / suits, all-in) -> run `experiments/observer_validation.py --fixture-dir tests/fixtures/pokernow`.
2. Longer v2 training (3 seeds to 300k, ~6.5 h) and a local best-response / exploitability estimate inside the abstraction.
3. Exact abstraction-error measurement on flop and turn subgames.
4. A better-calibrated rollout response model (the default model overfolds to large bets).

## Pre-real-data release candidate (Phase 80)

Branch `claude/live-observer`, Phases 41-80. Full suite: **783 passed, 0
skipped** (slow included, Python 3.13); clean venvs: 3.11 core-only 760
passed / 12 skipped (Streamlit), 3.12 all extras 773 passed.

| status | component | evidence (this run) |
| --- | --- | --- |
| VALIDATED (synthetic) | live observer infrastructure: test-session recorder, stored / recompute replay, annotation, privacy-preserving export, dataset roles, metrics, debugger, calibration v2, Retina geometry | tests; replay reproduces recorded sessions exactly |
| VALIDATED (synthetic / derived) | tracker plausibility rules under injected failures and perturbations | `docs/observer_failure_modes.md`, `docs/observer_robustness.md` |
| VALIDATED | observer performance and stability | 3 FPS within budget at p99; 60-min soak with flat RSS |
| **BLOCKED** | real PokerNow recognition accuracy | 1 real **tuning** frame, 0 validation / held-out frames |
| VALIDATED | decision pipeline fault handling, golden end-to-end outputs, determinism / resume | `tests/test_decision_faults.py`, `test_golden_e2e.py`, `test_determinism_audit.py` |
| VALIDATED | Monte Carlo error bars | equity coverage 0.89-0.99; rollout 0.95 mean (warning below 500 samples) |
| PARTIALLY VALIDATED | v2 abstraction | river error measured earlier; flop / turn exploitability 1.4-9.6 BB/hand in exact subgames -> flop / turn solver lookups capped at low confidence |
| PARTIALLY VALIDATED | rollouts | held-out agreement with exact river equilibria 0.61, optimistic, over-bet; response-model alternatives not adopted |
| EXPERIMENTAL | trained HU strategy | release promoted to NATIVE seed 0 @ 2M (final-trust project): +50..+117 bb/100 vs the 200k release in cross-play with matching 3-seed v2 confidence data; not an equilibrium, not GTO |

Release strategy (since the final-trust project):
`results/strategy/holdem_v2_native_seed0_2m.npz` (native, 2M) with its
SHA-bound v2 confidence table; the 200k and 100k pairs are archived
previous release. Rows above that were measured with the 100k release say
so (gate acceptance 19.5% at 100k vs 18.0% at 200k).
Gate thresholds: unchanged; added coded downgrade reasons
`OFF_TREE_TRANSLATION`, `ILLEGAL_SIZE_MASS`, `STREET_ABSTRACTION_ERROR`.
