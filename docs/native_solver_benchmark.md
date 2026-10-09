# Native solver benchmark (Phases 35, 94)

Machine: Apple M3 Pro (12 cores, 18 GB), macOS 15.5 arm64, Apple clang 21,
Python 3.12.6. Config: release `HoldemSolverConfig:v2:733e52f1d1014e2e7973`.
Data: `results/benchmarks/native_mccfr_v1.json` (3 trials, each in a fresh
process; medians reported). Command:
`python experiments/benchmark_native_solver.py`.
Python baseline details: `docs/native_solver_baseline.md`.

## Headline

| | Python (reference) | Native (C++) | ratio |
|---|---|---|---|
| warm training (≥10k iterations in) | **22.0 it/s** | **837 it/s** | **38.1x** |
| cold start (first 1k iterations) | 28.4 it/s | 1,250 it/s | 44x |
| RSS after the run | 924 MB (10k its) | 186 MB (100k its) | — |

Warm speedup is the adoption metric (target ≥5x, stretch ≥10x, excellent
≥20x): **38x measured end-to-end wall clock**, not a microbenchmark — the
timed loop is `train()` on the full release game, infoset table populated.

## Native throughput by block (trial 0; other trials within a few %)

| block | it/s | infosets after |
|---|---|---|
| 0–1k | 1,250 | 54,987 |
| 1k–5k | 935 | 89,628 |
| 5k–10k | 844 | 103,216 |
| 10k–20k | 873 | 114,009 |
| 20k–50k | 822 | 125,424 |
| 50k–100k | 833 | 131,896 |

Late-stage (3 parallel seeds, 250k–300k each): 770–810 it/s per process —
no meaningful thermal or table-growth degradation (Phase 110 check; the
aggregate over 3 workers was ~2,300 it/s sustained for 6.6 minutes).

## Checkpoint / export at 132k infosets

| operation | time | size |
|---|---|---|
| native checkpoint save | 0.09 s | 36.6 MB |
| native checkpoint load | 0.03 s | — |
| strategy artifact export | 0.84 s | 2.2 MB |

Checkpointing is negligible against training (a 300k run checkpoints 7
times ≈ 0.7 s total).

## Memory (Phase 46)

Per training process at 300k iterations / ~140k infosets: ~450 MB RSS
(nodes ~15 MB + feature caches; the Python solver reaches ~1 GB by 50k).
Rough bytes/infoset: ~130 B of table + cache overhead amortized.

## What was NOT adopted

* `-ffast-math` — never enabled; reproducibility depends on IEEE order.
* FP contraction — **disabled** (`-ffp-contract=off`); clang's default FMA
  fusion changed betting arithmetic in the last ulp and broke cross-backend
  bit parity (found by the parity suite, Phase 8).
* `-march=native`, custom hash maps, manual traversal stacks — not needed:
  at 38x with `std::unordered_map` and plain recursion, further
  architecture churn fails the maintainability trade-off (Phases 39/40/108).

## Projected training cost at measured late-stage throughput (Phase 95)

| experiment | sequential | 3 parallel workers |
|---|---|---|
| 100k, one seed | ~2 min | — |
| 300k × 3 seeds | ~20 min | **6.6 min (measured)** |
| 1M × 3 seeds | ~65 min | ~25 min |

(The Python baseline needed ~4 h per 300k seed; the committed v2 lineage
was trained at ~9 it/s in a 4-core container, ~9 h per 300k seed.)
