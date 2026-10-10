# Final release comparison and promotion (final-trust project, Phases 44-46)

Machine-readable: `results/validation/final_release_comparison.json`.

## Decision

**PROMOTED**: `results/strategy/holdem_v2_native_seed0_2m.npz` (native
backend, seed 0, 2,000,000 iterations, SHA-256 `f1860de7…`) with its
SHA-bound **SolverConfidence v2** table
(`holdem_v2_native_seed0_2m_confidence.npz`, `ef5c3223…`, seeds 0-2,
recent movement 1.5M→2M). The previous release (v2 seed 0 @ 200k + v1
table) is preserved untouched as `PREVIOUS_RELEASE_STRATEGIES["v2@200k"]`.

Status: **EXPERIMENTAL**. Imperfect-recall abstraction; restricted-LBR
lower bound ≥ ~100 bb/100 in the full game; not an equilibrium; not GTO.

## Why (all Phase-45 criteria met)

| criterion | evidence |
|---|---|
| exact validation | native MCCFR validated DIRECTLY on 7 exact games (ratio 1.04 vs Python; `native_direct_exact_validation.md`) |
| three comparable seeds | 2M × seeds 0-2, one lineage, exact resumes |
| matching confidence table | v2 schema, built from those seeds at 2M, SHA- and iteration-bound |
| methodology validated | v2 held-out risk-coverage ≥ v1 at equal accepted risk (`solver_confidence_v2.md`) |
| accepted-state risk | held-out: regret 0.123 vs 0.119 bb (±3%), wrong-action 3.0% vs 3.3% (better); mature slice 0.067 bb / 1.6% |
| cross-play | 2M beats the 200k release +50..+117 bb/100 (CIs exclude 0); 2M seeds pairwise ±5 |
| training maturity | seed disagreement median 0.617→0.466; canonical EV uncertainty 1.91→0.80 bb; consequential states 24→5 |
| sanity | premiums call jams ~100%, 72o folds 99%, AA 4-bets; no negative off-policy result; 14/14 postflop spots visited |
| artifact integrity | content SHA verified; bound-table mismatch tests green; golden e2e unchanged |

## Coverage (visit-weighted, after lookup-time street caps)

| street | accept | low | reject → rollout |
|---|---|---|---|
| preflop | 34.0% | 17.6% | 48.4% |
| flop | 0% (capped) | 72.5% | 27.6% |
| turn | 0% (capped) | 76.7% | 23.3% |
| river | 53.0% | 25.2% | 21.8% |

(Old release with v1 table: 8.9% / 16% / 19% / 19% accepts; the gain comes
from the validated movement-horizon fix plus training maturity, at
equal-or-better held-out accepted risk. Flop/turn stay capped at LOW —
measured abstraction error is never erased by stable training.)

## What promotion does NOT claim

Cross-play wins are abstract-game head-to-head; the LBR floor shows every
candidate (old and new) remains heavily exploitable in the full game by a
range-tracking adversary; rollout/heuristic fallbacks carry their own
documented biases (`final_rollout_audit.json`). See
`docs/solver_status_final.md` for the complete evidence hierarchy.
