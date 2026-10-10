# Native solver project — progress / recovery record

Recovery sources: git history, this file, results/benchmarks/*,
results/validation/native_*.json, docs/native_solver*.md.

## FINAL STATE (all phases complete)

- branch claude/live-observer; starting HEAD b398c4e (fast-forwarded from
  d43ae59 at session start; local was 43 behind, 0 ahead).
- Native backend: cpp/ (C++17, pybind11, scikit-build-core), module
  poker_alpha_native, wrapper poker_alpha/native/. pip install ./cpp.
- Correctness: exhaustive evaluator parity; 20k+ state mechanics/key
  parity; bitwise 1,000-iteration random-tape parity; exact checkpoint
  resume; corruption rejection; ASan+UBSan self-test clean; CI green on
  Linux + macOS arm64 native jobs.
- Performance: 38.1x warm (837 vs 22 it/s, median of 3 trials, M3 Pro) —
  results/benchmarks/native_mccfr_v1.json.
- Training: fresh native seeds 0,1,2 -> 300k (6.6 min) -> 1M (~21 min),
  milestone checkpoints local-only (gitignored), logs committed.
- Studies: results/validation/native_training_300k.json,
  native_training_1m.json, crossplay_1m_vs_300k.json,
  gate_acceptance_v2_native_{300k,1000k}.json.
- Release decision: 1M seed0 is the recommended candidate
  (results/strategy/candidates/native_1m/); NOT auto-promoted (gate
  acceptance 16.2% < 18.0% under the fixed earlier=10k movement signal;
  preflop seed disagreement still severe). Release pointer unchanged:
  holdem_v2_seed0_200k.npz.
- Docs: native_solver{,_design,_baseline,_benchmark,_architecture_audit}.md,
  release_status.md and solver_validation.md updated; README section added.

## Remaining known work (not blocking)

- Owner decision on promoting the 1M candidate.
- Optional: recalibrate the movement signal with a consecutive-checkpoint
  construction (would need re-validation against exact-game error).
- Optional LBR lower-bound estimator (Phase 69) — deliberately not started.
