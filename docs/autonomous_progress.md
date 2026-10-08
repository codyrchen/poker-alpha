# Autonomous progress (solver quality and release)

Recovery file for a fresh session. Machine-readable twin:
`results/validation/autonomous_progress.json`.

- branch: `claude/solver-quality-and-release` (from `8335ea2`)
- baseline: `POKERALPHA_SKIP_SLOW=1 pytest` = 527 passed, 5 skipped, 196 s (2026-10-08)
- current phase: 35/36 (v2 retraining in flight; Phase 33/34/35-river runs finishing)
- last completed phase: 32

## Persistent inputs
Training checkpoints of the locked config (`HoldemSolverConfig:v1:2e23c098860f7894c871`),
seeds 0/1/2 at 10k / 100k / 300k: `/home/user/pa_ckpt/locked_seed{s}_it{n}.npz`
(container-local, ~25-28 MB each, NOT in git). Regenerate with
`python experiments/holdem_mccfr_validation.py --locked-config --seed S --milestones 1000,3000,10000,30000,100000 --ckpt-dir DIR --out DIR/locked_seedS.jsonl`
then `--resume DIR/locked_seedS_it100000.npz --milestones 300000` (~3.5 h per seed).
Everything later phases need from them is extracted into committed JSON
(see below) so they are not irreplaceable.

## Experiments running
- Phase 33: `python experiments/phase33_reduced_games.py` -> results/validation/reduced_holdem_v1.json (log /home/user/pa_ckpt/phase33.log). Deterministic; rerun if lost.
- Phase 34: `python experiments/phase34_abstraction_error.py` -> results/validation/abstraction_error_v1.json (log phase34.log).
- Phase 35 river buckets: `python experiments/phase35_river_buckets.py` -> results/validation/abstraction_error_river_pct.json (log phase35rp.log).
- Phase 35 v2 retraining (V2_CONFIG = HoldemSolverConfig:v2:733e52f1d1014e2e7973: legal sizing + compact
  encoder with 20 exact river-percentile buckets), seeds 0/1/2, started 2026-10-08 ~19:00 UTC, ~4 h:
  `python experiments/holdem_mccfr_validation.py --v2-config --seed S --milestones 1000,5000,10000,30000,100000 --ckpt-dir /home/user/pa_ckpt/v2b --out /home/user/pa_ckpt/v2b/v2_seedS.jsonl`
  (restarted at 5k with the exact 3x faster river ranking, 19:25 UTC)
  complete when each jsonl has an `"iterations": 100000` row; resume with
  `--resume /home/user/pa_ckpt/v2b/v2_seedS_it<last>.npz --milestones <remaining>`.
- Ablation (stopped at 30k on purpose): sizing-only v2 (LEGAL_SIZING_CONFIG) checkpoints in /home/user/pa_ckpt/v2/legal_seedS_it{1000,5000,10000,30000}.npz.

## Results so far
- Phase 32: see docs/solver_validation.md. No rule/utility/MCCFR/averaging bug; preflop strategy noise-dominated; sub-minimum b33 sizes; facing-jam key collision.

- Phase 33 (partial): MCCFR on the exact reduced preflop game converges (exploitability 0.58 -> 0.007 BB, 100 -> 300k it., 3 seeds agree).
- Phase 34A: compact encoder strategy exploitable by 9.24 BB (dry board) / 2.24 BB (four-flush) in the raw river game vs 0.006 / 0.019 for raw; 20 river percentile buckets -> 0.57 BB (dry).
- Phase 37A: seed disagreement predicts true error (Spearman 0.61); thresholds reject >= 0.5, low >= 0.2; movement reject >= 0.5, low >= 0.1; visits not predictive (20-visit floor).
- Phase 39: clairvoyant toy game matches the analytic solution at 50/100/200% pot.
- Rollouts converge to closed forms within 3 SE.

## Open bugs
- abstract b33 sizes below NLHE minimums (semantic issue of v1 abstraction)
- betting-context collision: BTN facing a jam after any open size shares one key

## Next exact action
Wait for v2 training (100k x 3 seeds); then analyse with experiments/phase29_analysis.py-style tooling for v2, build v2 artifact + confidence table, compare to v1, decide primary config, then docs/release (Phase 40).
