# Release notes — final solver-trust project (2026-10)

## New release strategy

* `holdem_v2_native_seed0_2m.npz` — native backend, 2,000,000 iterations,
  seed 0 (previous 200k and 100k releases preserved in place). Beats the
  previous release +50..+117 bb/100 in duplicate cross-play (CIs exclude 0).
* Still **experimental**: imperfect-recall abstraction, not an equilibrium,
  not GTO; a newly implemented restricted-LBR lower bound shows ≥ ~100
  bb/100 full-game exploitability for every strategy in this family.

## Confidence methodology v2

* One change: the movement signal now measures the recent checkpoint
  interval instead of 10k→final (which was measuring training distance, not
  instability). Thresholds, other signals, street caps unchanged.
* Calibrated on a new 8,880-row exact-game dataset with exact EV-regret
  targets; held-out: ~1.7x the coverage at equal accepted risk, lower
  wrong-action rate. v2 tables SHA-bind to their strategy artifact.
* Effect with street caps: preflop 34% / river 53% visit-weighted accepts
  (was 9/19%); flop and turn remain capped at LOW.

## New measurement capability

* Direct native exact-game validation (tabular adapter): native MCCFR ≡
  Python statistically on all 7 exact games.
* Restricted LBR (validated: Spearman 0.986 vs exact exploitability on
  reduced games) — the project's first full-game exploitability evidence.
* Policy-vs-value studies: ~94% of preflop seed disagreement is
  near-equivalent mixing (EV-flat); the consequential residue (jam/3-bet
  defense) is identified and still shrinking with training.

## Training

* Seeds 0-2 extended 1M → 2M (final milestone; 3M evaluated and rejected —
  remaining differences are dominated by the abstraction floor).

## What did not change

* Game rules, encoder, config signature, Python reference solver, rollout
  defaults (audited, kept), observer (read-only, still blocked on
  independent real data), historical v1 confidence semantics.
