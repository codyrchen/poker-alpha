# Solver status — authoritative summary (final-trust project, Phase 63)

Single source of truth for the solver side of PokerAlpha as of the
final-trust project. Where any older document disagrees, THIS file and the
result JSONs it cites win. Evidence hierarchy at the bottom.

## Official release

| | |
|---|---|
| strategy | `results/strategy/holdem_v2_native_seed0_2m.npz` — native backend, seed 0, **2,000,000 iterations**, SHA-256 `f1860de75526111f9cd321e8f7f19fa94620a9856a930e5511432e5c64362da2` |
| confidence | `holdem_v2_native_seed0_2m_confidence.npz` — **SolverConfidence v2** (recent movement 1.5M→2M, seeds 0-2, SHA-bound), `ef5c32238f173d6ec3dcdd8c68c8cae6f406ea863bcee3797e2102764945b422` |
| config | `HoldemSolverConfig:v2:733e52f1d1014e2e7973` (unchanged since Phase 40) |
| status | **EXPERIMENTAL** abstract-game strategy — imperfect recall, no equilibrium guarantee, not GTO |
| previous releases | v2@200k and v2@100k preserved in place (`PREVIOUS_RELEASE_STRATEGIES`) |
| promotion evidence | `docs/final_release_comparison.md` / `results/validation/final_release_comparison.json` |

## Best research candidate

The release IS the best candidate (2M). Further training: **NO-GO**
(`post_2m_training_decision.json`) — 2M vs 1M head-to-head mostly within
noise, remaining differences dominated by the abstraction floor. Hard
ceiling 3M was not needed.

## Native backend

READY. 38x warm speedup (837 vs 22 it/s, M3 Pro); bitwise random-tape
parity with the Python reference; **direct** exact-game validation (7 games,
exploitability ratio 1.04 vs Python at matched iterations); exhaustive
evaluator parity; ASan/UBSan clean; Linux + macOS CI. Unchanged during this
project except the tabular exact-validation adapter (`cpp/src/tabular.cpp`).

## Confidence methodology

**v2 adopted** (`docs/solver_confidence_v2.md`): identical signals and
thresholds to v1, movement redefined to the consecutive-recent horizon,
tables SHA-bound to their strategy. Held-out exact-game calibration
(8,880 rows, 9 games): +75% coverage at equal accepted EV regret, lower
wrong-action rate; movement predicts EV regret at NO horizon (|rho|≤0.05);
visits is the strongest regret predictor (AUC 0.74); seed disagreement the
best wrong-action/L1 predictor. v1 tables still load with v1 semantics.

## Street status

| street | abstraction | training |
|---|---|---|
| preflop | card mapping lossless | instability is ~94% near-equivalent MCCFR mixing (measured by action EVs); 7→5 consequential jam/3-bet-defense states remain, shrinking with training |
| flop | exploitability median 3.7 bb / max 5.1 bb on 7 exact subgames → capped at LOW | covered by shared signals |
| turn | median 5.1 bb / max 8.7 bb on 7 exact subgames → capped at LOW | covered by shared signals |
| river | pct20 buckets: median 0.42 bb on 6 exact subgames (4–16x better than v1 encoder) | 53% visit-weighted accept |

Authoritative abstraction numbers: `docs/abstraction_error_summary.md`.

## Rollout fallback

20/36 exact-river root agreement, over-bet bias; MDF response models
evaluated and NOT adopted (held-out); fast depth default (13x faster);
MC error bars calibrated (SE ratio 0.86–1.27), <500-sample warning.
`results/validation/final_rollout_audit.json`.

## Exploitability status

* Exact full-game exploitability: NOT COMPUTED (infeasible).
* Restricted-LBR lower bound (validated: Spearman 0.986 vs exact on reduced
  games, no upper violations): **≥ ~100 bb/100 for every candidate**
  (200k: 123±25, 300k: 127±24, 1M: 103±19). This is the honest headline
  limitation: the strategy family is far from unexploitable in the full
  game, and that floor is abstraction-driven.
* Exact reduced games: native MCCFR converges to ~0 exploitability.

## Observer relationship

Unchanged by this project. Read-only; real PokerNow recognition remains
**BLOCKED ON INDEPENDENT DATA** (1 tuning frame, 0 held-out frames).

## Known limitations

1. Full-game exploitability floor ≥ ~100 bb/100 (restricted LBR) — an
   abstraction property, not fixable by training.
2. Flop/turn abstraction error measured and material → permanent LOW caps.
3. Confidence v2 calibrated on preflop/river-shaped exact games at 3k–30k
   maturity; production application extrapolates in maturity (checked on
   the most mature slice: risk improves) and in street (flop/turn rely on
   caps + shared signals).
4. Jam/3-bet-defense cluster: 5 canonical states keep significant cross-
   seed EV uncertainty (up to ~15 bb at jams).
5. Rollout fallback over-bets vs exact river equilibria (16/36 spots).
6. Collision signal remains a heuristic.

## Next research question

The only solver-side lever left with measured upside is the **abstraction
itself** (richer flop/turn buckets or perfect-recall-preserving context),
which would attack the LBR floor and the street caps — a new config/lineage
decision, out of scope for autonomous promotion. Everything else solver-side
is resolved or explicitly bounded. The project-level next step is
independent real PokerNow observer data.

## Evidence hierarchy (Phase 58)

* **Strong**: exact reduced-game exploitability and EV regret
  (`native_direct_exact_validation.json`, `confidence_calibration_dataset.json`);
  held-out confidence calibration (`confidence_signal_quality.json`).
* **Medium**: restricted-LBR lower bounds (`lbr_holdem.json`); cross-seed
  value agreement (`policy_vs_value_disagreement_*.json`); duplicate-deal
  cross-play (`native_training_{1m,2m}.json`).
* **Weak/sanity**: canonical-hand inspection, fixed scripted opponents
  (`offpolicy_*.json`), policy-frequency aesthetics. Never promote on these.
