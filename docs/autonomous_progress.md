# Autonomous progress (solver quality and release)

Recovery file for a fresh session. Machine-readable twin:
`results/validation/autonomous_progress.json`.

- branch: `claude/solver-quality-and-release` (from `8335ea2`)
- baseline: `POKERALPHA_SKIP_SLOW=1 pytest` = 527 passed, 5 skipped, 196 s (2026-10-08)
- current phase: 35 (v2 legal-sizing retraining; 33/34/37A runs in flight)
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
- Phase 37A: `python experiments/phase37_calibration.py` -> results/validation/solver_gate_calibration.json (log phase37cal.log).
- Phase 35 v2 retraining (HoldemSolverConfig:v2:1eca9c95607b7a7fb008, legal sizing), seeds 0/1/2:
  `python experiments/holdem_mccfr_validation.py --legal-sizing-config --seed S --milestones 1000,5000,10000,30000,100000 --ckpt-dir /home/user/pa_ckpt/v2 --out /home/user/pa_ckpt/v2/legal_seedS.jsonl`
  checkpoints `/home/user/pa_ckpt/v2/legal_seedS_it{1000,5000,10000,30000,100000}.npz`; complete when each jsonl has an `"iterations": 100000` row.
  Resume a killed seed with `--resume /home/user/pa_ckpt/v2/legal_seedS_it<last>.npz --milestones <remaining>`.

## Results so far
- Phase 32: see docs/solver_validation.md. No rule/utility/MCCFR/averaging bug; preflop strategy noise-dominated; sub-minimum b33 sizes; facing-jam key collision.

## Open bugs
- abstract b33 sizes below NLHE minimums (semantic issue of v1 abstraction)
- betting-context collision: BTN facing a jam after any open size shares one key

## Next exact action
Phase 33: build exact reduced Hold'em games (experiments/phase33_reduced_games.py).
