# Solver evidence inventory (final-trust project, Phase 2)

Authoritative mapping of every major solver claim to its committed evidence,
with staleness resolved by reading the result files — not memory. Candidate
identity: `results/validation/final_release_inventory.json` (all 12
artifacts load-tested, SHAs recorded).

| claim | evidence file | verdict |
|---|---|---|
| River abstraction error measured exactly; pct20 cuts it 4–16x | `results/validation/abstraction_error_river_pct.json`, `abstraction_error_v1.json` (6 × 52-card river subgames, exact CFR+) | VALID |
| Flop abstraction error "not measured exactly" (release_status component table) | `results/validation/flop_abstraction_v1.json` | **STALE — WRONG.** The flop study is complete: 7 exact reduced flop subgames (fixed turn/river samples, reduced bet menu), all four encoders. compact_pct20 exploitability median 3.7 bb / max 5.1 bb per subgame (pot 5 BB); EV error vs raw ≤ 0.13 bb. |
| Turn abstraction error "not measured exactly" | `results/validation/turn_abstraction_v1.json` | **STALE — WRONG.** 7 exact reduced turn subgames complete. compact_pct20 exploitability median 5.1 bb / max 8.7 bb; EV error ≤ 0.66 bb. |
| "flop results partial (1 of 7)" (historical commit message) | same file | superseded — all 7 rows complete in the committed JSON |
| MCCFR correct on exact reduced games (exploitability ~0.007 at 300k, 3 seeds) | `results/validation/reduced_holdem_v1.json`, `docs/solver_validation.md` | VALID (Python backend) |
| Native ≡ Python bitwise on shared random tape (1,000 iterations) | `tests/test_native_mccfr_parity.py`, CI | VALID |
| Native exact reduced-game validation | previously **transitive only** | superseded by the direct study this project adds (`results/validation/native_direct_exact_validation.json`) |
| Gate calibration: seed disagreement Spearman 0.61 vs true error; visits ≈ no signal (−0.06); movement 0.33 | `results/validation/solver_gate_calibration.json` (3 exact games × 2 milestones, 1,888 pooled infosets) | VALID but narrow: movement there = 3k→30k; the production tables use 10k→final, a different horizon. Recalibrated by this project. |
| Release = v2 seed 0 @ 200k + 200k confidence table | `poker_alpha/solver_config.py`, `tests/test_release_strategy.py`, inventory | VALID |
| 1M native candidate beats release +52..+56 bb/100 (CIs exclude 0); not promoted | `results/validation/native_training_1m.json`, `crossplay_1m_vs_300k.json`, `docs/release_status.md` | VALID |
| Gate acceptance 18.0% (200k) / 18.1% (300k) / 16.2% (1M) visit-weighted | `results/validation/gate_acceptance_v2_{200k,native_300k,native_1000k}.json` | VALID; the 1M drop is driven by movement-vs-10k (see movement_signal_study.json, this project) |
| Rollouts agree with exact river equilibria in 20/36 root spots; overbet bias | `results/validation/rollout_validation_v1.json` | VALID (as of its run; re-examined in final_rollout_audit.json) |
| MDF-style response models NOT adopted (held-out: no improvement) | `results/validation/rollout_response_v2.json` (96-game calibration/held-out split) | VALID |
| Deep (showdown) rollouts: 6/14 action flips, median EV shift 1.9 bb, 13x slower; fast mode kept default | `results/validation/rollout_depth.json` | VALID |
| MC error bars calibrated (bias/RMSE/SE/coverage at 5 sizes) | `results/validation/mc_error_calibration.json` | VALID |
| Exact best response on full game INFEASIBLE; abstract BR not well-defined (imperfect recall); LBR feasible and recommended | `results/validation/best_response_feasibility.json` | VALID — LBR is implemented by this project (Phase 17/18) |
| Preflop noise-dominated; all/most preflop first-action keys gated out | `preflop_audit_{v1,v2}.json`, gate acceptance files | VALID at 200k–1M |
| Native backend 38x, bitwise parity, CI green | `results/benchmarks/native_mccfr_v1.json`, `docs/native_solver_benchmark.md` | VALID |
| Real PokerNow recognition unvalidated (1 tuning frame, 0 held-out) | `docs/release_status.md`, observer docs | VALID — BLOCKED ON INDEPENDENT DATA (unchanged by this project) |

## Corrections applied by this project

1. `docs/release_status.md` component table: flop/turn abstraction rows now
   point at the completed 7-board exact studies instead of "not measured
   exactly" (fixed in Phase 15 along with
   `docs/abstraction_error_summary.md`, the single authoritative view).
2. Native exact-game validation upgraded from transitive to direct
   (Phase 3).
3. The movement confidence signal's horizon dependence is quantified rather
   than asserted (Phase 5).
