# Solver confidence v2 (final-trust project, Phases 20-25)

Schema: `pokeralpha.solver_confidence/v2`. Code:
`poker_alpha/decision/solver_gate.py` (`ConfidenceTable`, schema 2),
builder: `experiments/phase37_build_confidence.py --schema 2`.
Calibration evidence: `results/validation/confidence_calibration_dataset.json`
(8,880 rows, 9 exact games, exact EV-regret targets) and
`confidence_signal_quality.json` (leave-one-game-out).

## What changed vs v1 — exactly one thing

**The movement signal's horizon.** v1 measured L1(10k → final): at 1M that
marks 71% of keys above the REJECT threshold while correlating *less* with
instability as training matures (it measures distance traveled). v2
measures **recent movement**: L1(previous mature milestone → final), e.g.
750k → 1M, recorded in `meta["movement_from"/"movement_to"]`.

Everything else is deliberately unchanged: signals (visits, movement, seed
disagreement, collision), thresholds (visits 20 floor; seed disagreement
0.2 low / 0.5 reject; movement 0.1 low / 0.5 reject; collision 3/7 low),
decision rule, street caps (flop/turn LOW), pathological flags, lookup-time
downgrades. v1 tables keep loading with v1 semantics (schema recorded).

## Why (held-out, exact-game evidence)

Signal quality vs exact EV regret (pooled Spearman; LOGO stable):

| signal | rho vs EV regret | rho vs L1 | AUC (regret > 2% pot) |
|---|---|---|---|
| visits | **−0.35** | −0.18 | **0.74** |
| seed disagreement | +0.23 | **+0.57** | 0.63 |
| movement (any horizon) | ≤ 0.05 | 0.10–0.20 | ≈ 0.45–0.49 |

Risk-coverage with the SAME thresholds (held-out games):

| rule | coverage | accepted EV regret | wrong-action |
|---|---|---|---|
| v1 (historical movement) | 9.8% | 0.119 bb | 3.3% |
| **v2 (recent movement)** | **16.5%** | 0.123 bb | 3.0% |
| no movement at all | 23.4% | 0.170 bb | 3.0% |

v2 delivers ~1.7x the coverage at equal accepted risk; dropping movement
entirely buys more coverage at visibly higher accepted regret, so movement
is kept as a conjunction filter (it still correlates with L1 and costs
nothing). Street-specific thresholds were evaluated and NOT adopted: the
same rule yields comparable pot-relative accepted regret preflop (1.1%)
and river (1.7%); abstraction risk per street is already handled by the
unchanged LOW caps.

The v1 visit floor is *reinforced* by this calibration: visits turned out
to be the strongest EV-regret predictor (the historical "visits don't
predict error" finding was about L1, which the Phase-8 studies show is the
wrong target).

## Table format v2 additions (Phase 24)

* `meta["schema"] = 2`, `movement_mode = "recent"`, `movement_from/_to`;
* `meta["strategy_sha256"]` — the bound strategy artifact's SHA-256; the
  provider refuses a bound table whose hash does not match the loaded
  artifact, and any v2 table whose `final` differs from the artifact's
  iteration count (`tests/test_confidence_v2.py`). No silent 1M-strategy /
  200k-table pairings.

## Backward compatibility (Phase 25)

v1 tables load unchanged (schema 1, historical-movement semantics) and are
never reinterpreted under v2 rules. The current release's v1 table remains
exactly as calibrated and shipped. Historical results are not restated.

## Known limitations

* Calibration games are preflop- and river-shaped; flop/turn confidence
  relies on the street caps plus the shared signals.
* `collision` remains a heuristic (unchanged, un-recalibrated).
* Coverage numbers above are dataset coverage; production coverage depends
  on the strategy's own signal distribution (reported per candidate in
  `results/validation/gate_acceptance_*.json`).
* Confidence is not a probability of correctness; it is a calibrated
  accept/low/reject rule on measured proxies.
