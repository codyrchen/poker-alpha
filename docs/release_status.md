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
| PARTIALLY VALIDATED | compact abstraction (v2) | river error measured on exact subgames and cut 4-16x; flop/turn error not measured exactly |
| PARTIALLY VALIDATED | gate thresholds | calibrated on exact games (seed disagreement vs true error, Spearman 0.61), applied to the full abstraction by extrapolation |
| PARTIALLY VALIDATED | decision response models | rollouts disagree with exact river equilibria in 16 / 36 spots (they overbet); recommendations change with the assumed opponent model |
| PARTIALLY VALIDATED | screen observer on synthetic images | synthetic fixtures; annotation validation and metrics harness ready |
| EXPERIMENTAL | trained HU strategy (v2, 3 seeds x 100k) | 10 / 10 sanity checks per seed; not converged; preflop noise-dominated (all 169 BTN first-action keys gated out) |
| EXPERIMENTAL | range and opponent modelling | beliefs under heuristic priors; no ground-truth accuracy |
| EXPERIMENTAL | real PokerNow recognition | heads-up preset aligned on 1 real annotated frame; all fields correct on it at 0.8x-3x, but the same frame was used for tuning, so accuracy is **not measured** (post-RC, `docs/observer.md`) |

## Release solver

| | |
| --- | --- |
| config | `HoldemSolverConfig:v2:733e52f1d1014e2e7973` (compact encoder + 20 river percentile buckets, legal NLHE sizing, 100 BB, raise cap 3, external-sampling MCCFR, uniform averaging) |
| artifact | `results/strategy/holdem_v2_seed0.npz` (seed 0, 100,000 iterations; checksum in `release_candidate.json`) |
| confidence table | `results/strategy/holdem_v2_seed0_confidence.npz` |
| use at decision time | gate: 19% of visit-weighted decisions accepted, 16% low confidence, 65% rejected -> rollout / heuristic |

## Performance (4-vCPU container, CPython 3.13, pure Python + NumPy)

Offline: v2 training ~9 it/s per process (~3 h per seed to 100k); v1 ~30
it/s. Online (median, `results/validation/latency_benchmark.json`):

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
| EXPERIMENTAL | trained HU strategy | release unchanged (seed 0, 100k); 200k is measurably stronger in cross-play, promotion awaits the owner's decision |

Release strategy: unchanged (`results/strategy/holdem_v2_seed0.npz`, 100k).
Gate thresholds: unchanged; added coded downgrade reasons
`OFF_TREE_TRANSLATION`, `ILLEGAL_SIZE_MASS`, `STREET_ABSTRACTION_ERROR`.
