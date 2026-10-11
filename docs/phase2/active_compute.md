# Active compute allocation (Phase II, Program 2)

**Question.** Under a fixed compute budget, does steering external-sampling
MCCFR toward strategically unresolved regions converge faster than standard
sampling?

**Answer: NULL.** Importance-corrected prioritized root-chance sampling, with
four transparent priority signals and an exact best-response oracle, did not
produce a replicated improvement over standard ES-MCCFR at equal node
traversals on Leduc or on the reduced preflop Hold'em game. The one signal
that was consistently ahead on one game (policy entropy on reduced preflop,
about +2 to +7 % lower exploitability) is the same size as what an
*information-free* fixed random reweighting produces (-13 % to +5 %). It also
costs more wall time per node to score. It is not a usable improvement.

## Method

`poker_alpha/research/active_compute.py`. The root chance outcome (the deal)
is sampled from `q = (1 - eps) p + eps * score / sum(score)` instead of the
natural distribution `p`. Every regret and average-strategy increment made
under that outcome is multiplied by `p / q`, so each increment keeps its
expectation (importance sampling; `eps = 0.5` bounds weights by 2). Scores are
recomputed every 50 iterations.

| signal | score of a deal |
|---|---|
| `positive_regret` | summed cumulative positive regret of the infosets reachable under it |
| `scarcity` | mean of 1 / (1 + visits) over those infosets |
| `entropy` | mean current-strategy entropy over those infosets |
| `random_fixed` | fixed random score (control: reweighting with no information) |
| `oracle_br_gain` | exact per-deal best-response gain of the current average (Leduc only) |

With a uniform signal the solver is bit-identical to `MCCFRSolver` in
strategy and node count (`tests/test_active_compute.py`), so any difference
is the allocation.

Compute is counted as node visits (`CountingGame`, one per visited node).
Budgets: Leduc 2e4 to 5e6 nodes, reduced preflop 2e4 to 2e6. Metric: exact
exploitability, evaluation excluded. Discovery used seeds 0-9. Confirmation
used independent seeds 100-119 for the two signals that looked best.

## Results (relative exploitability reduction vs MCCFR at equal nodes, per budget)

| game / signal | discovery (10 seeds) | confirmation (20 seeds) |
|---|---|---|
| Leduc scarcity | -5 % .. +11 % (classified POSITIVE) | -7 % .. +3 % (NULL) |
| Leduc entropy | -12 % .. +4 % | -6 % .. +5 % |
| Leduc random_fixed (control) | -9 % .. +5 % | -13 % .. -6 % |
| Leduc oracle BR gain | -18 % .. +4 % | not rerun |
| reduced preflop scarcity | -6 % .. +2 % (NEGATIVE) | +1 % .. +7 % (POSITIVE) |
| reduced preflop entropy | -1 % .. +6 % | +2 % .. +7 % |
| reduced preflop random_fixed (control) | -2 % .. +3 % | -5 % .. +3 % |

Per-budget classifications flipped between discovery and confirmation for
three of four signal/game pairs. That is what multiple comparisons over 7-8
budgets x 3 signals x 2 games produce under a null effect. Even the exact
best-response oracle did not help, which suggests that in these games the
deal-level allocation is not the bottleneck. ES-MCCFR's variance comes from
opponent-action and later-chance sampling inside each deal, which this scheme
does not touch.

Artifacts: `results/phase2/active_compute.json`,
`results/phase2/active_compute_confirm.json`.

## Systems finding produced by this program

At equal nodes, plain MCCFR on the reduced preflop game used about 4x the
wall time of the prioritized variants. Profiling traced it to
`Game.sample_chance` rebuilding every deal state on each call. The reduced
games now cache their chance outcomes and draw with the identical single
`rng.choice` call. Same-seed strategy digest unchanged; MCCFR throughput on
the reduced preflop game went from 3,227 to about 12,400 it/s (3.8x).
The equal-*time* curves in the two artifacts predate this fix for reduced
preflop and must not be used. The equal-node comparisons are unaffected.

## Limitations

Root-level (deal) prioritization only. Deeper chance and opponent-action
sampling are untouched. Two small exact games. The Python solver timings are
not representative of the native backend.

```bash
python experiments/phase2_active_compute.py --workers 10
python experiments/phase2_active_compute.py --seed-base 100 --seeds 20 \
  --methods mccfr,prio:scarcity,prio:entropy,prio:random_fixed --out results/phase2/active_compute_confirm.json
```
