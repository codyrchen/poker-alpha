# v2 release candidates (preserved, NOT the release)

`RELEASE_STRATEGY` is still `results/strategy/holdem_v2_seed0.npz` (seed 0,
100k). Nothing here is loaded by default. Same config for every file:
`HoldemSolverConfig:v2:733e52f1d1014e2e7973`.

| file | what | why kept |
| --- | --- | --- |
| `holdem_v2_seed0_200k.npz` + `_confidence.npz` | **recommended release candidate** and its matching gate table (seeds 0-2 at 200k, movement vs 10k — same construction as the 100k release table) | candidate |
| `holdem_v2_seed0_300k.npz` | seed 0 at 300k; **no compatible confidence table** (seeds 1-2 stop at 200k) | strongest in cross-play; needed if seeds 1-2 are ever extended |
| `holdem_v2_seed1_200k.npz`, `holdem_v2_seed2_200k.npz` | seeds 1-2 at 200k | reproduce seed disagreement / cross-play without the 44 MB checkpoints |
| `MANIFEST.json` | SHA-256 and provenance of these files, plus every training checkpoint (not committed: 12-45 MB each, container only) with resume / regeneration commands | |
| `training_logs/` | JSONL milestone rows of the v2 runs | provenance |

Comparison: `results/validation/release_candidates_v2.json`
(`python experiments/compare_release_candidates.py --dir results/strategy/candidates/v2_extension`).

To adopt the 200k candidate (owner decision): copy both 200k files to
`results/strategy/`, point `RELEASE_STRATEGY` in `poker_alpha/solver_config.py`
(and `STRATEGY` in `poker_alpha/ui/app.py`, `DEFAULT_STRATEGY` in
`poker_alpha/platform_demo.py`) at it, re-run the test suite and regenerate
the golden file only if the diff is the intended one.
