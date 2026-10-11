# Native systems research (Phase II, Programs 7, 21, 23)

**Result: release-config native MCCFR training is 1.96x faster with
bit-identical results.** Median throughput went from 858 to 1,678 it/s over
alternating A/B runs on identical Release flags. The strategy digest is
identical on every run. Python/native shared-tape parity, checkpoint
resume, exhaustive 5-card evaluator parity, a new 255-bucket river
percentile parity test on flush-heavy boards and the ASan+UBSan self-test
all pass on the final build.

Artifacts: `results/phase2/native_profile.json`,
`native_optimization_trials.json`, `native_memory.json` and
`native_correctness_stress.json`.

## 7A Profile (before)

macOS `sample` at 1 ms over 10 s of release-config training. The build was
RelWithDebInfo with the Release flags `-O3 -g`, because pybind11 strips
Release builds.

| function | self | inclusive |
|---|---|---|
| `pa::evaluate` | 43.1 % | |
| `std::sort` internals | 30.2 % | |
| `MCCFR::get_or_create` (std::unordered_map) | 10.8 % | 10.8 % |
| `FeatureCache::river_values` | 6.7 % | **69.5 %** |
| `FeatureCache::compute` | 1.4 % | 81.3 % |
| `HoldemGame::utility` | | 1.3 % |

Diagnosis: the v2 abstraction's river percentile buckets. For every new
river board the feature cache evaluated up to 1,081 holdings and sorted
them. External sampling draws a fresh board on almost every chance branch,
and there are 2.6M boards, so the 4,096-board cache rarely hits. More than
two thirds of training time went to building tables that were used once or
twice. Game logic, regret matching and hashing were minor.

## Changes (each benchmarked, all kept)

| trial | change | median it/s (old to new) | speedup | digest |
|---|---|---|---|---|
| T1 | deduplicated river table: a holding's value depends only on canonical card ids (exact card for the single flush-capable suit, rank otherwise). Evaluate each distinct pair once with its multiplicity. Store distinct values plus cumulative counts, so there is no 1,081-element sort | 853.9 to 1,362.0 | 1.60x | identical |
| T2 | evaluator split into count accumulation and a shared finish step; board histogram computed once per table | 858.6 to 1,459.3 | 1.70x | identical |
| T3 | flat open-addressing infoset index (splitmix64, linear probing, load <= 1/2) replaces `std::unordered_map` | 858.2 to 1,678.3 | **1.96x** | identical |

5, 4 and 3 alternating repetitions respectively. The standard deviation is
below 1 % of the median in every arm. Peak RSS at 30k iterations fell from a
median of about 263 MB to about 220 MB, because the compact river tables are
smaller.

Why T1 is exact: a 5-card board has at most one suit with three or more
cards, and only cards of that suit can contribute to a flush. Two holdings
whose cards have the same canonical ids therefore have the same 7-card value.
The percentile counts "holdings below" and "holdings equal", which are
identical whether computed from the full sorted list or from distinct values
with multiplicities. Multiplicities sum to 1,081 (C(47,2)).

## Rejected or not attempted (NO-GO)

* **SIMD / NEON evaluator.** After T1-T3, evaluation is about 100-300
  branch-heavy scalar calls per new board. A vector evaluator would need a
  table-driven design and a new exhaustive parity proof. Payoff is bounded
  by evaluate's remaining share, which was not re-profiled after T3.
* **Bigger river-board cache.** Hit rate is structural: boards rarely repeat
  under external sampling. Measured cache occupancy stays near 2k of 4k.
* **`-ffast-math` and float32 training.** These break bit parity with the
  Python reference, a project invariant. The regret table is about 20 MB,
  so memory is not a reason to change.

## 7D Key representation

The packed uint64 key uses 38 bits, and the packing is a bijection with the
canonical key string. This is pinned by the existing parity tests. The
all-ones key cannot occur, which the flat index uses as its empty marker.
With identity hashing the structured bitfields would cluster, so the index
uses the splitmix64 finalizer.

## 21 / 23 Correctness stress and memory scale

* **Checkpoint stress.** 100,000 continuous iterations versus 10 x 10,000
  with save and reload in between: identical state hash.
* **Shared-tape parity.** 10,000 iterations of Python versus native on one
  random tape: zero value mismatches, identical tape position (4,484,738
  uniforms consumed), bit-identical average strategy over 102,404 infosets.
  This is 10x the previous longest parity run.
* **Evaluator.** Exhaustive parity over all 20,358,520 six-card and all
  133,784,560 seven-card hands, plus 200k suit-permutation checks, with zero
  mismatches (`evaluator_stress.json`).
* **Memory and scale** (1M-iteration run, seed 11):

| iterations | infosets | peak RSS (MB) | segment it/s |
|---|---|---|---|
| 10k | 102,604 | 120.7 | 1,719 |
| 100k | 132,684 | 131.5 | 1,590 |
| 400k | 141,962 | 131.5 | 1,629 |
| 1M | 145,955 | 131.5 | 1,480 |

  The table saturates. The abstraction's reachable key space is about
  146k infosets, using about 147 bytes per infoset of table, so RSS is
  flat. Throughput falls about 14 % as the table fills. A linear fit gives
  about 236 bytes of RSS per infoset. Reaching 8/16/32 GB would require
  about 36M/72M/145M infosets, which is **unreachable at this
  abstraction**. The iteration extrapolation in the artifact is labelled
  as meaningless. The bottleneck is CPU in feature computation, not
  memory.

Reproduce:

```bash
pip install ./cpp
python -m pytest tests/test_native_*.py
python experiments/validate_native_solver.py --hands 200 --iters 50
python experiments/phase2_native_stress.py --parts scale,checkpoint,parity
```
