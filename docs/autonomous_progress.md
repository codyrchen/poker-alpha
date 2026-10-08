# Autonomous progress (solver quality and release)

Recovery file for a fresh session. Machine-readable twin:
`results/validation/autonomous_progress.json`.

- branch: `claude/solver-quality-and-release` (from `8335ea2`)
- baseline: `POKERALPHA_SKIP_SLOW=1 pytest` = 527 passed, 5 skipped, 196 s (2026-10-08)
- current phase: complete — Phase 40 release candidate
- last completed phase: 40

## Persistent inputs
Training checkpoints of the locked config (`HoldemSolverConfig:v1:2e23c098860f7894c871`),
seeds 0/1/2 at 10k / 100k / 300k: `/home/user/pa_ckpt/locked_seed{s}_it{n}.npz`
(container-local, ~25-28 MB each, NOT in git). Regenerate with
`python experiments/holdem_mccfr_validation.py --locked-config --seed S --milestones 1000,3000,10000,30000,100000 --ckpt-dir DIR --out DIR/locked_seedS.jsonl`
then `--resume DIR/locked_seedS_it100000.npz --milestones 300000` (~3.5 h per seed).
Everything later phases need from them is extracted into committed JSON
(see below) so they are not irreplaceable.

## Experiments running
none. v2 checkpoints (seeds 0-2 at 1k/5k/10k/30k/100k): /home/user/pa_ckpt/v2b/v2_seedS_itN.npz
(container-local; regenerate with the --v2-config command in README). Sizing-only ablation at 30k: /home/user/pa_ckpt/v2/.

## Results so far
- Phase 32: see docs/solver_validation.md. No rule/utility/MCCFR/averaging bug; preflop strategy noise-dominated; sub-minimum b33 sizes; facing-jam key collision.

- Phase 33 (partial): MCCFR on the exact reduced preflop game converges (exploitability 0.58 -> 0.007 BB, 100 -> 300k it., 3 seeds agree).
- Phase 34A: compact encoder strategy exploitable by 9.24 BB (dry board) / 2.24 BB (four-flush) in the raw river game vs 0.006 / 0.019 for raw; 20 river percentile buckets -> 0.57 BB (dry).
- Phase 37A: seed disagreement predicts true error (Spearman 0.61); thresholds reject >= 0.5, low >= 0.2; movement reject >= 0.5, low >= 0.1; visits not predictive (20-visit floor).
- Phase 39: clairvoyant toy game matches the analytic solution at 50/100/200% pot.
- Rollouts converge to closed forms within 3 SE.

- Phases 33-40: see docs/solver_validation.md and docs/release_status.md; release config v2, artifact results/strategy/holdem_v2_seed0.npz.

## Open bugs
- none known. Known limitations: preflop strategy noise-dominated (gated out), rollout response model overfolds to big bets, real PokerNow accuracy unmeasured.

## Next exact action
None required. Recommended future work: v2 seeds to 300k; exact flop/turn abstraction error; real annotated PokerNow fixtures.
