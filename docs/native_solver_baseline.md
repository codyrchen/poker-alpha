# Python MCCFR baseline (Phase 1)

Machine: Apple M3 Pro (12 cores, 18 GB), macOS 15.5 arm64, Python 3.12.6,
NumPy 2.5.2. Config: `HoldemSolverConfig:v2:733e52f1d1014e2e7973` (release),
seed 0. Data: `results/benchmarks/mccfr_python_baseline.json`,
profile: `results/benchmarks/mccfr_python_profile.txt`.
Command: `python experiments/profile_mccfr_baseline.py --iterations 50000`.

## Throughput (single process)

| block | it/s | ms/it | infosets after |
|---|---|---|---|
| 0 – 1k (cold) | 28.4 | 35.3 | 53,395 |
| 1k – 5k | 23.7 | 42.3 | 88,598 |
| 5k – 10k | 21.7 | 46.1 | 101,875 |
| 10k – 20k | 21.0 | 47.7 | 114,038 |
| 20k – 50k (warm) | **21.0** | **47.7** | 126,182 |

Peak RSS ≈ 1.0 GB (solver tables + feature/equity caches). The historical
"~9 it/s" figure was the previous 4-core container; **21 it/s warm** is the
baseline the native speedup target is measured against on this machine.

## Work per iteration (counted over 1k iterations)

~329 decision nodes, ~243 terminal visits... precisely: 328.9 decision,
129.6 chance, 243.3 terminal node visits and 570.2 `next_state` calls per
iteration (two traversals). 53k infosets discovered in the first 1k
iterations; discovery slows sharply after ~10k.

## Hot path (cProfile, 2k iterations, % of self time)

| % | function |
|---|---|
| 35.9 | `poker/evaluator.py::evaluate_best_codes` |
| 15.5 | `abstraction/features.py::_river_values` (river percentile tables) |
| 7.0 | `builtins.sorted` (river value tables, key sorting) |
| 5.0 | dict `.get` (memo tables) |
| 3.0 | `evaluator._straight_high_from_mask` |
| 2.4 | `mccfr._traverse` |
| 1.3 | `features._is_current_nuts` |
| 1.3 | `holdem.sample_chance` |
| 1.3 | `holdem.next_state` |

The evaluator plus river-percentile machinery is ~60% of runtime; betting
replay and encoder string assembly share most of the rest. This is exactly
the profile the native port attacks: all of it runs in C++ with cached
features and no string keys.

## Native comparison

See `results/benchmarks/native_mccfr_v1.json` /
`docs/native_solver_benchmark.md` for the measured native numbers
(median-of-trials, same config, same machine).
