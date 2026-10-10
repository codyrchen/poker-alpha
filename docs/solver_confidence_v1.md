# Solver confidence v1 — frozen specification (final-trust project, Phase 4)

Exact behavior of the production confidence methodology as implemented at
commit `334587d`, recorded before any redesign. Machine-readable twin:
`results/validation/solver_confidence_v1_frozen.json`.
Code: `poker_alpha/decision/solver_gate.py`; table construction:
`experiments/phase37_build_confidence.py`.

## Table construction (`pokeralpha.solver_confidence/v1` npz)

Per infoset key of the primary seed's final checkpoint (keys with
`strategy_sum > 0`):

| signal | definition |
|---|---|
| `visits` | non-updating-player visits of the primary seed at the final milestone (`strategy_sum.sum()`) |
| `movement` | L1 between the primary seed's average strategy at the **earlier** and **final** checkpoints. Production tables use `earlier = 10,000` for the v2 lineage (release 200k table: 10k→200k; native tables: 10k→300k, 10k→1M). NaN if the key is absent earlier. |
| `seed_disagreement` | mean pairwise L1 of the final average strategies across seeds 0-2, over the seeds where the key exists; NaN if only one seed has it |
| `collision` | range of the 0..7 strength ladder among corpus states (20,000 seeded hands) that map to the key, / 7; NaN below 5 corpus members |

`pathological`: keys flagged by the Phase-32 strategy audit (v1 lineage
only; empty in every v2 table).

## Gate (`gate(stats, thresholds, pathological)`)

Thresholds in production (`GateThresholds.calibrated()` reading
`results/validation/solver_gate_calibration.json`):

| threshold | value | origin |
|---|---|---|
| reject_visits_below | 20 | documented sanity floor (visits had ~no predictive value: Spearman −0.06) |
| low_visits_below | 20 | max(calibrated 5, floor 20) |
| reject_seed_disagreement | ≥ 0.5 | exact-game calibration |
| low_seed_disagreement | ≥ 0.2 | exact-game calibration |
| reject_movement | ≥ 0.5 | exact-game calibration (movement there = 3k→30k) |
| low_movement | ≥ 0.1 | exact-game calibration |
| low_collision | ≥ 3/7 | heuristic (no exact reference for the full abstraction) |

Decision order: missing stats or `visits <= 0` → REJECT `UNSEEN_STATE`.
Otherwise collect reasons; any reject-level reason ⇒ REJECT (low reasons
appended); else any low-level reason ⇒ LOW; else ACCEPT. NaN seed
disagreement AND NaN movement adds `NO_STABILITY_DATA` (low). Collision and
pathological flags are low-only. 

## Lookup-time (spot-level) rules — can only lower the status

* `OFF_TREE_TRANSLATION`: observed sizes mapped onto the abstract tree → downgrade.
* `ILLEGAL_SIZE_MASS`: ≥ 0.2 of solver mass removed as below-minimum sizes → downgrade; all mass illegal → miss (`OUTSIDE_ABSTRACTION`).
* `STREET_ABSTRACTION_ERROR`: flop (street 1) and turn (street 2) lookups of
  `CompactHoldemEncoder` are capped at LOW (exact subgame studies: lifted
  compact strategies 1.4–9.6 bb exploitable vs ~0.02 raw).

## Calibration provenance

`solver_gate_calibration.json`: 3 exact games (reduced preflop + 2 river
subgames) × milestones 3k/30k, 1,888 pooled infosets. Spearman vs true
strategy L1 error: seed disagreement **0.61**, movement 0.33, visits −0.06.
Note the calibration's movement is a *consecutive* 3k→30k horizon, while
production tables measure 10k→final — at 1M this is a 100x longer horizon
than anything calibrated, the central v1 design flaw quantified in
`results/validation/movement_signal_study.json`.

## Known v1 consequences (measured before this project)

* Visit-weighted acceptance: 18.0% (200k), 18.1% (native 300k), 16.2%
  (native 1M) — the drop at 1M is caused by the movement signal's fixed
  10k baseline, not by less stable strategies (seed disagreement improved
  0.617 → 0.523 median).
* Preflop first-action keys: essentially all rejected at every milestone.
