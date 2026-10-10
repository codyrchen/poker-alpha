# Final solver-trust project — progress / recovery record

Recover from: git, this file, results/validation/final_solver_progress.json,
committed reports. Do not ask the user.

- branch: claude/live-observer; starting HEAD: 177236f (clean, == origin)
- native backend v0.1.0 installed and working; 1.2G of training checkpoints
  for native seeds 0-2 at 10k..1M present locally (results/native_training/,
  gitignored) — needed for movement features and any 2M resume.
- release: v2_200k_seed0 (+v1 table); recommended candidate: native_1m_seed0.
- Phase 1 DONE: results/validation/final_release_inventory.json (12 artifacts, no problems)
- Phase 2 DONE: docs/solver_evidence_inventory.md. KEY RESOLUTION: flop AND
  turn abstraction error ARE fully measured (7 exact subgames each,
  flop_abstraction_v1.json / turn_abstraction_v1.json); release_status
  claim "not measured exactly" is stale -> fix in Phase 15.
- Evidence notes: gate calibration (seed dis 0.61 best, visits ~0 signal);
  LBR feasibility: GO recommended (best_response_feasibility.json);
  rollout response models NOT ADOPTED (keep); rollout depth: fast default kept.
- NEXT: Phase 3 native direct exact validation (tabular-tree adapter in cpp/),
  then confidence studies.
