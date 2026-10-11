# Abstraction science (Phase II, Program 5)

Four controlled studies on exact domains. Reproduce all four with
`python experiments/phase2_abstraction_science.py --workers 10` (about 5 min).

## 5A Capacity / error frontier

`results/phase2/abstraction_capacity.json`. The domain is the Program 1
river subgames: 9 boards x 3 seeds, 48 combos per player, 400 CFR+
iterations. K equal-count strength cells per player. The error is the exact
exploitability of the lifted strategy in the raw game, in chips of a
10-chip pot.

| K per player | infosets | table bytes | exploitability |
|---|---|---|---|
| 1 | 14 | 608 | 3.829 |
| 2 | 28 | 1,216 | 1.073 |
| 3 | 42 | 1,824 | 0.595 |
| 4 | 56 | 2,432 | 0.347 |
| 6 | 84 | 3,648 | 0.198 |
| 8 | 112 | 4,864 | 0.145 |
| 12 | 168 | 7,296 | 0.108 |
| 16 | 224 | 9,728 | 0.089 |
| 24 | 336 | 14,592 | 0.062 |
| 48 (raw) | 672 | 29,184 | 0.004 |

Error falls roughly as `K^-1.5` up to K = 8, then flattens: K = 24 still
carries 15x the raw error. Throughput is flat in K (about 2,000-2,200 it/s),
because this solver's cost is per hand, not per infoset. **Error-guided
refinement versus uniform, at equal infosets:** across 540 matched points
from the refinement paths, adaptive is better at 86 % of points, with a
median uniform/adaptive error ratio of 1.36 and a mean of 1.91. This
corroborates Program 1 along the whole frontier, not only at one budget.

## 5B Action abstraction: exact cost of a restricted bet menu

`results/phase2/action_abstraction.json`. The reference is menu D (25 %,
50 %, 75 %, 100 %, 150 %, 200 % pot, plus all-in) for both players. The
*restricted loss* is the exact value a player gives up when limited to a
coarser menu while the opponent keeps D. Both games are solved exactly, with
600 iterations, 32 combos, 9 boards x 3 seeds. No measured loss was negative
beyond the solver error bound of 0.008 chips.

| menu (bets) | decision nodes | infosets | it/s | OOP loss mean / max | IP loss mean / max |
|---|---|---|---|---|---|
| A (100 %) | 8 | 256 | 3,841 | 0.014 / 0.151 | 0.018 / 0.295 |
| B (50, 100 %) | 14 | 448 | 2,154 | 0.012 / 0.151 | 0.011 / 0.179 |
| C (33, 75, 150 %) | 22 | 704 | 1,339 | 0.007 / 0.106 | 0.004 / 0.051 |
| D (reference) | 40 | 1,280 | 701 | 0 | 0 |

Mean losses are small, 0.1-0.2 % of the pot, but the tail is not. A single
pot-size bet can cost 3 % of the pot in some spots. Menu C, the same
fractions as the release's postflop menu, cuts the mean loss to about half
of B's and the worst case by 2-3x, at 1.6x the tree size and 1.6x the time
per iteration. Each additional size buys less: the step from C to D adds
82 % more nodes to remove the last 0.004-0.007 chips of mean loss.

## 5C Perfect vs imperfect recall (Leduc)

`results/phase2/recall_ablation.json`. Each abstraction defines an abstract
Leduc on the same tree. CFR+ solves it, and the lifted strategy's exact
exploitability is measured in real Leduc at 10, 30, 100, 300, 1,000 and 3,000
iterations.

| abstraction | recall | infosets | exploitability at 10 / 100 / 3,000 iterations |
|---|---|---|---|
| none | perfect | 288 | 0.610 / 0.013 / 0.000 |
| J = Q in round 1 only | perfect | 282 | 0.650 / 0.135 / 0.131 |
| J = Q in both rounds | perfect | 192 | 1.154 / 0.745 / 0.729 |
| round-1 history to its raise count | imperfect | 180 | 0.560 / 0.371 / 0.397 |
| round-1 history forgotten | imperfect | 72 | 0.479 / **0.859** / **0.939** |
| round 2 keeps only pair/above/below | imperfect | 228 | 0.918 / 0.432 / 0.444 |

Findings:

* **More training makes the forgetful abstraction worse.** With round-1
  history forgotten, real-game exploitability *rises* from 0.48 to 0.94 as
  CFR+ trains. The "raise count only" abstraction also gets worse after
  100 iterations. Under imperfect recall the CFR average need not converge
  to anything good in the real game. This is a concrete demonstration that
  "train longer" can be counterproductive for the release's
  imperfect-recall encoder class.
* **Infoset count is a poor proxy for abstraction quality.** Merging J and
  Q only in round 1 removes 6 of 288 infosets (2 %) and costs 0.13 chips,
  more than a third of the cost of the raise-count abstraction, which
  removes 37 % of infosets.
* Time per iteration is the same for all abstractions here (about 50 ms),
  because the Python solver walks the same tree. The memory saving does not
  buy speed in tree-walking solvers.

## 5D Collision dispersion: does within-bucket heterogeneity predict error?

`results/phase2/collision_dispersion.json`. 216 cells (K = 4, 27 instances).
Each cell's *policy dispersion*, the reach-weighted TV of the member hands'
raw-equilibrium policies around their mean, is compared with its measured
error contribution (best-response gain) in the abstract solution:

| signal | Spearman vs cell error | top-cell agreement |
|---|---|---|
| raw-policy dispersion | 0.50 | 26 % |
| equity spread | 0.45 | 56 % |
| shared-action regret (action values) | **0.93** | 78 % |

Members whose raw-optimal *policies* differ are not necessarily the cells
that cost the most. What matters is how much value is lost by forcing one
action on the cell, which is a value-based measure. This explains why
Program 1's `equity_spread` signal was good but not best. It also bears on
the release, whose collision component measures strength-ladder spread
(structural, like equity spread) and covers only 1.4 % of keys
(Program 10). A value-based collision measure is the better design if it
is ever rebuilt.

## Limitations

The river studies are single-street. 5C is one game and one solver. 5B's
menus all include all-in, and the raise cap is 2. None of these numbers are
full-Hold'em measurements.
