# Autonomous progress (pre-real-test run, Phases 41-80)

Recovery file for a fresh session. Machine-readable twin:
`results/validation/autonomous_progress.json`. Reconstruct from these files
and `git log`, not from chat history.

- branch: `claude/live-observer`
- start commit: `8a35231674b97982015454c2c649695713b43697`
- previous run (Phases 26-40): see Git history of this file and
  `docs/release_status.md`, `docs/solver_validation.md`.

## Data-discipline rule
One real PokerNow frame exists: `tests/fixtures/pokernow/raw/hu_preflop_0001.png`.
It is a TUNING fixture (layout, OCR and card recognizer were fitted on it).
Never tune further visual parameters on it; never count transforms of it as
independent data (label: DERIVED FROM TUNING FRAME — NOT INDEPENDENT VALIDATION).

## Persistent inputs
- v2 checkpoints (seeds 0-2 at 1k/5k/10k/30k/100k): `/home/user/pa_ckpt/v2b/v2_seedS_itN.npz`
  (container-local, ~43 MB each at 100k; not in git).
- v1 locked checkpoints: `/home/user/pa_ckpt/locked_seedS_itN.npz`.

## Long-running experiments
### Phase 65 v2 extension (started at the beginning of this run)
- question: is v2 still training-limited at 100k? Does 3x iterations shrink
  seed disagreement / raise gate acceptance, or is the remaining error structural?
- commands (each a separate background process, 1 core each, ~9 it/s):
  - seed 0 (primary): `python experiments/holdem_mccfr_validation.py --v2-config --seed 0 --resume /home/user/pa_ckpt/v2b/v2_seed0_it100000.npz --milestones 150000,200000,250000,300000 --ckpt-dir /home/user/pa_ckpt/v2c --out /home/user/pa_ckpt/v2c/v2_seed0.jsonl` (~6.2 h)
  - seeds 1, 2 (controls): same with `--seed S --resume .../v2_seedS_it100000.npz --milestones 150000,200000` (~3.1 h each)
  - logs: `/home/user/pa_ckpt/v2c/seedS.log`
- expected: seed disagreement falls (roughly 1/sqrt(T) if sampling-noise dominated); preflop still noisy.
- checkpointing: every 50k iterations (`--resume` from the latest if killed).
- stop condition: seed 0 at 300k, seeds 1-2 at 200k. No 1M run.
- analysis: experiments/v2_extension_analysis.py (Phase 65) -> results/validation/holdem_training_v2_300k.json

## Phase log
(see JSON twin for the authoritative list)
