# Native solver benchmark (Phases 35, 94; updated in Phase II)

All numbers are **training throughput of external-sampling MCCFR on the
release configuration** (`HoldemSolverConfig:v2:733e52f1d1014e2e7973`).
They are not a speedup for every operation. Strategy lookup, decision
support and evaluation tools do not use this training loop.

Machine: Apple M3 Pro (12 cores, 18 GB), macOS arm64, Apple clang, Python
3.12.6.

## The three measurements, kept separate

| | what it measured | result | artifact |
|---|---|---|---|
| **Phase I** | Python reference vs the first native port | 22.0 vs **837** it/s warm median, **38.1x** | `results/benchmarks/native_mccfr_v1.json` |
| **Phase II A/B experiment** | old native build vs optimized native build only, identical Release flags, alternating processes, 30,000 iterations from a fresh table | median **858 → 1,678** it/s, **1.96x**, identical strategy digest | `results/phase2/native_optimization_trials.json` |
| **Current fresh benchmark** | Python reference vs the optimized native build, same script and protocol as Phase I | 21.9 vs **1,625** it/s warm median, **74.2x** | `results/benchmarks/native_mccfr_v2.json` |

The A/B figure (1,678 it/s) and the fresh benchmark figure (1,625 it/s) are
**different experiments** and are not interchangeable. The A/B runs average
over iterations 0-30k of a fresh table. The benchmark's "warm" number is
the 50k-100k block of a 100k run, with a larger table. Each number is
stated against its own baseline.

## Current benchmark protocol (`native_mccfr_v2.json`)

Command:

```bash
python experiments/benchmark_native_solver.py --out results/benchmarks/native_mccfr_v2.json
```

* 3 trials per backend, seeds 0-2, each in a fresh process. Medians are
  reported.
* Python: 10,000 iterations per trial. Native: 100,000 iterations per trial.
* Warm throughput is the last milestone block (Python 5k-10k, native
  50k-100k), with the infoset table populated. The timed call is `train()`
  on the full release game: end-to-end wall clock, not a microbenchmark.

| | Python (reference) | Native (C++) | ratio |
|---|---|---|---|
| warm training (median) | **21.9 it/s** | **1,625 it/s** | **74.2x** |
| cold start (first 1k iterations, median) | 29.6 it/s | 2,286 it/s | 77x |
| RSS after the run | about 850 MB (10k its) | about 102-131 MB (100k its) | n/a |

Native throughput by block (trial 0):

| block | it/s | infosets after |
|---|---|---|
| 0-1k | 2,286 | 54,987 |
| 1k-5k | 1,786 | 89,628 |
| 5k-10k | 1,622 | 103,216 |
| 10k-20k | 1,706 | 114,009 |
| 20k-50k | 1,610 | 125,424 |
| 50k-100k | 1,625 | 131,896 |

The infoset counts equal Phase I's trial 0 at every block. Training
trajectories did not change, only their speed.

Checkpoint and export at 132k infosets: save 0.09 s (36.6 MB), load 0.02 s,
strategy artifact export 0.83 s (2.2 MB).

## Phase II optimization (`docs/phase2/native_systems.md`)

Profile before (macOS `sample`, -O3 -g build): 69.5 % of training CPU was
inside `FeatureCache::river_values`. Each new river board evaluated up to
1,081 holdings and sorted them. External sampling rarely revisits a board,
so the 4,096-board cache seldom hit. Retained changes, each A/B benchmarked
with an identical strategy digest:

| change | A/B speedup (cumulative) |
|---|---|
| deduplicated river percentile construction: evaluate each distinct canonical holding once (exact card for the only flush-capable suit, rank otherwise), cumulative-count table, no 1,081-value sort | 1.60x |
| evaluator board-count reuse: board histogram accumulated once per table | 1.70x |
| flat open-addressing infoset index (splitmix64, linear probing) replacing `std::unordered_map` | 1.96x |

### Not adopted (NO-GO)

* **`-ffast-math`**: never enabled. FP contraction stays disabled
  (`-ffp-contract=off`), because bit parity with the Python reference
  depends on IEEE operation order.
* **float32 training for the release**: rejected. It changes numerics and
  breaks parity, and the regret table is only about 20 MB.
* **SIMD / NEON evaluator**: not adopted. The remaining evaluation is
  branch-heavy scalar code and would need a new table-driven evaluator
  with a new exhaustive parity proof.
* **A bigger river-board cache**: not useful. Boards almost never repeat
  under external sampling; occupancy stays near 2k of the 4k limit.

## Memory (`results/phase2/native_memory.json`)

One 1M-iteration native run (seed 11): the table saturates at about
146,000 infosets. Peak RSS is flat at **131.5 MB** from 100k iterations on,
with about 147 bytes of table per infoset. Segment throughput falls about
14 % as the table fills (1,719 → 1,480 it/s). Memory is bounded by the
abstraction's key space, not by iterations.

## Correctness of the optimized build

* Python vs native on a shared random tape for **10,000 iterations**:
  bit-identical iterate values and average strategy
  (`results/phase2/native_correctness_stress.json`).
* 100,000 continuous iterations equal 10 x 10,000 with checkpoint
  save/load between chunks (same file).
* Evaluator equal to the Python reference on **all 20,358,520 six-card and
  all 133,784,560 seven-card hands**, plus 200,000 suit-permutation checks
  (`results/phase2/evaluator_stress.json`); exhaustive 5-card parity as before.
* ASan + UBSan native self-test passes.

## Historical projections (Phase 95, at Phase I throughput)

| experiment | sequential | 3 parallel workers |
|---|---|---|
| 100k, one seed | ~2 min | n/a |
| 300k × 3 seeds | ~20 min | 6.6 min (measured) |
| 1M × 3 seeds | ~65 min | ~25 min |

At Phase II throughput these roughly halve. The 2M → 4M extension of three
seeds ran at about 1.3k it/s per process under concurrent load.
