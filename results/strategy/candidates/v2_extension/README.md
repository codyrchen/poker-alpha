# v2 release candidates (preserved)

The 200k pair below was **promoted**: identical copies are the release
files `results/strategy/holdem_v2_seed0_200k.npz` and
`holdem_v2_seed0_200k_confidence.npz` (`RELEASE_STRATEGY`). The previous
release is `results/strategy/holdem_v2_seed0.npz` (100k). Nothing in this
folder is loaded by default. Same config for every file:
`HoldemSolverConfig:v2:733e52f1d1014e2e7973`.

| file | what | why kept |
| --- | --- | --- |
| `holdem_v2_seed0_200k.npz` + `_confidence.npz` | **promoted to release** — identical to the files in `results/strategy/` — and its matching gate table (seeds 0-2 at 200k, movement vs 10k — same construction as the 100k release table) | candidate |
| `holdem_v2_seed0_300k.npz` | seed 0 at 300k; **no compatible confidence table** (seeds 1-2 stop at 200k) | strongest in cross-play; needed if seeds 1-2 are ever extended |
| `holdem_v2_seed1_200k.npz`, `holdem_v2_seed2_200k.npz` | seeds 1-2 at 200k | reproduce seed disagreement / cross-play without the 44 MB checkpoints |
| `MANIFEST.json` | SHA-256 and provenance of these files, plus every training checkpoint (not committed: 12-45 MB each, container only) with resume / regeneration commands | |
| `training_logs/` | JSONL milestone rows of the v2 runs | provenance |

Comparison: `results/validation/release_candidates_v2.json`
(`python experiments/compare_release_candidates.py --dir results/strategy/candidates/v2_extension`).

Adopted: `RELEASE_STRATEGY` now points at the 200k file (app and demo load
it through that constant); `tests/test_release_strategy.py` pins it. The
golden end-to-end outputs did not change.
