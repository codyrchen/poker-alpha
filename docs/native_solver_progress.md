# Native solver project — progress / recovery record

Autonomous recovery file. If context is lost: recover from git history,
this file, results/benchmarks/*, results/validation/native_*.json. Do not
ask the user.

## State (updated after 3-seed 300k training)

- branch claude/live-observer; starting HEAD b398c4e; pushed through fbfc119+
- Phases 0-55 complete: build (cpp/, pip install ./cpp), parity (evaluator
  exhaustive, 20k+ state corpus, bitwise 1000-iteration tape parity),
  checkpoints (exact resume, corruption rejection), artifact export,
  doctor, CI (Linux+macOS native jobs GREEN), clean-install test, ASan/UBSan
  self-test clean, --backend in holdem_mccfr_validation.py, multiseed runner.
- Python baseline: 21 it/s warm (results/benchmarks/mccfr_python_baseline.json)
- Native: ~770-800 it/s warm single process; 3 parallel seeds ~765 each.
- TRAINING DONE: 3 fresh native seeds (Plan B; no resumable Python
  checkpoints exist) 0->300k, 6.6 min total.
  Checkpoints: results/native_training/seed{0,1,2}/v2_native_seed{s}_it{m}.npz
  (gitignored); logs: results/native_training/v2_native_seed{s}.jsonl.
  140k infosets/seed; top-2000 L1 (250k->300k) ~0.045.
- Full fast suite: 811 passed, 12 skipped. CI green.

## In flight

- experiments/benchmark_native_solver.py (3 trials) -> results/benchmarks/native_mccfr_v1.json
- next: experiments/native_300k_study.py (artifacts, confidence table w/
  earlier=10k, gate acceptance, canonical preflop, crossplay vs release)

## Next actions (in order)

1. finish benchmark -> docs/native_solver_benchmark.md, commit
2. run native_300k_study.py -> results/validation/native_training_300k.json
3. 1M go/no-go per Phase 65 (cost ~25 min for 3 seeds in parallel; decide
   on movement/disagreement trends from the study)
4. if GO: resume seeds to 1M (milestones 400k,500k,750k,1M), rerun study at 1M
5. release-candidate decision (Phase 100-102; promotion only if all gates met)
6. README/native docs finalization, full+slow suites, final report
