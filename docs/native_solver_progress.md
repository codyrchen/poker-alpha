# Native solver project — progress / recovery record

Autonomous recovery file. If context is lost, recover from: git history,
this file, results/benchmarks/native_solver_progress.json, committed
benchmark outputs, checkpoint manifests. Do not ask the user.

## Current state

- branch: claude/live-observer
- starting HEAD: b398c4e (after fast-forward from d43ae59; local was 43 behind, 0 ahead)
- phase: 1 (baseline profile running), 0 complete (audit written)
- active experiment: Python baseline profile, 50k iterations, release config, seed 0
  (experiments/profile_mccfr_baseline.py) — running in background
- machine: Apple M3 Pro, 12 cores, 18 GB, macOS 25.5.0 (arm64), Apple clang 21,
  Python 3.12.6 (.venv), numpy 2.5.2 (venv) / 2.4.1 (system). No cmake binary
  (will be a pip build dependency).
- test baseline: POKERALPHA_SKIP_SLOW=1 pytest -q -> 783 passed, 6 skipped (green)

## Key verified facts

- Release: V2_CONFIG (HoldemSolverConfig:v2:733e52f1d1014e2e7973),
  strategy holdem_v2_seed0_200k.npz + confidence (seeds 0-2 @200k).
- Candidates: seed0 {100k,200k,300k} artifacts; seed1,2 {200k}. 300k has NO
  confidence table. NO resumable checkpoints exist anywhere in the checkout
  (manifest lists 44 by SHA only, "container only"). => Phase 19/59: Python->
  native import is moot; fresh native training (Plan B) is the route.
- Python cold throughput ~26 it/s at iterations 0-200 on this machine
  (historical container: ~9 it/s); warm numbers in
  results/benchmarks/mccfr_python_baseline.json when the run finishes.
- ~327 decision nodes / iteration; max 6 actions per node.

## Plan (decided)

- cpp/ tree with pybind11 + scikit-build-core + CMake as a separate
  native component; poker_alpha/native/ Python wrapper; see
  docs/native_solver_design.md and docs/native_solver_architecture_audit.md.
- Packed uint64 infoset key rendering byte-identical canonical strings.
- Random-tape parity harness on both backends; xoshiro256** production RNG.
- Native checkpoint .npz written Python-side from arrays; artifact export
  reuses strategy_artifact.export_solver via shim.

## Exact next action

Wait for baseline profile to finish; write docs/native_solver_baseline.md;
commit Phase 0-1; then scaffold cpp/ build (Phase 3).
