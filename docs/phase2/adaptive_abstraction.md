# Adaptive abstraction refinement (Phase II, Program 1)

**Question.** Can *measured* strategic error decide where additional
abstraction capacity should go, and does that beat spending the same
capacity uniformly or at random?

**Answer (confirmed).** Yes, in the exact river domain tested. Splitting the
cell with the largest exact best-response gain lowers raw-game exploitability
by 24–31 % relative to a uniform partition with *identical* per-player cell
counts, and by 44–57 % relative to random splitting, on every configuration
tried (discovery, independent confirmation, higher capacity). The greedy
error-guided rule recovers 95–98 % of what an exhaustive one-step oracle
achieves. A solver-internal signal (cumulative positive regret) is a clear
**negative**: it is no better than random.

Artifacts: `results/phase2/adaptive_abstraction.json` (discovery, 30
instances), `adaptive_abstraction_confirm.json` (confirmation: new seeds,
3 new boards, 45 instances), `adaptive_abstraction_highcap.json` (K=8,
16 splits, 64 hands/player, 18 instances).

Reproduce:

```bash
python experiments/phase2_adaptive_abstraction.py              # discovery   (~2 min, 10 workers)
python experiments/phase2_adaptive_abstraction.py --confirm    # confirmation (~6 min)
python experiments/phase2_adaptive_abstraction.py --k0 8 --splits 16 --hands 64 --seeds 3 \
    --out results/phase2/adaptive_abstraction_highcap.json
pytest tests/test_river_solver.py tests/test_refinement.py
```

## Domain (1A)

52-card river subgames of the Phase 33 family: fixed board, explicit ranges
for both players, pot 10 / stack 20, bets 50 % + 100 % pot + all-in, raise
cap 2, out-of-position player acts first. Ranges are sampled (seeded) from
river-strength percentile bands (6 discovery boards, 3 additional
confirmation boards; 48 or 64 combos per player). This is the closest exact
domain to the Hold'em abstraction: real cards, real evaluator, real bet
tree; it is single-street, so a partition of the private hand is a
perfect-recall abstraction and the abstract game's CFR+ solution is an
equilibrium of the abstract game. Error is the exact exploitability of the
lifted strategy in the raw (unabstracted) subgame, in chips.

Solver: `poker_alpha/research/river_solver.py`, a vectorized range-vs-range
CFR+/CFR with per-player hand partitions. It reproduces
`poker_alpha.solvers.cfr_plus.CFRPlusSolver` on `RiverSubgame` to 1e-15
(`tests/test_river_solver.py`) and is ~30x faster per iteration on 8x8
ranges (and the gap grows with range size), which is what made a
multi-seed, multi-control study feasible (~180 exact solves per instance).

## Baseline abstraction (1B)

K = 4 equal-count strength-ordered cells per player (8 cells, 56 abstract
infosets on the 14-node betting tree, vs 672 raw infosets with 48 hands).
400 CFR+ iterations; the raw solve's exploitability (solver-error floor)
is 0.004 chips, baseline abstraction error ~0.31 chips. Memory of the
infoset tables is proportional to infoset count (2 x 8 bytes per action).

## Failure signals (1C)

Per player and cell (all exact; `poker_alpha/research/refinement.py`):

| signal | definition | needs a solve |
|---|---|---|
| `br_gain` | sum over member hands of (best-response cf value − profile cf value); the two players' totals add to 2 x exploitability, so it decomposes the measured error by cell | yes |
| `shared_action_regret` | one-step cost of the best *single* pure action for the cell vs each member hand's own best action, summed over the player's nodes | yes |
| `equity_spread` | max − min static equity vs the opponent's prior range among members | no |
| `positive_regret` | solver's final cumulative positive regret summed over the cell's infosets | yes (internal) |
| `random` | uniform random cell | no |

No combined score is used.

## Refinement (1D) and controls (1E)

Greedy: solve, score, split the top cell at its strength median, re-solve;
`splits` rounds. Controls with the same number of cells:

* `uniform_matched`: equal-count partition with exactly the per-player cell
  counts the `br_gain` arm ended with (identical infoset count);
* `uniform_symmetric`: equal-count with the total split evenly between players;
* `random`: the same number of random (player, cell) splits, 5 seeds;
* `oracle`: each round tries every possible split and keeps the best
  (upper bound on one-step greedy selection; 10–20x the compute).

## Results (1F–1H)

Exploitability (chips, mean over instances; lower is better; paired
bootstrap 95 % CIs of the difference vs each control):

| arm | discovery (30 inst.) | confirmation (45 inst.) | high capacity (18 inst.) |
|---|---|---|---|
| raw (floor) | 0.0040 | 0.0042 | 0.0046 |
| baseline K | 0.315 | 0.369 | 0.130 |
| random | 0.228 | 0.251 | 0.099 |
| uniform_matched | 0.140 | 0.146 | 0.081 |
| uniform_symmetric | 0.128 | 0.152 | 0.083 |
| adaptive `positive_regret` | 0.237 | 0.266 | 0.088 |
| adaptive `equity_spread` | 0.110 | 0.119 | 0.066 |
| adaptive `shared_action_regret` | 0.097 | 0.109 | 0.058 |
| **adaptive `br_gain`** | **0.097** | **0.110** | **0.056** |
| oracle | 0.094 | 0.099 | 0.054 |

Paired `br_gain` vs `uniform_matched`: discovery −0.043 chips
(CI −0.060..−0.028, 26/30 wins, −31 %); confirmation −0.036
(CI −0.048..−0.024, 39/45 wins, −24 %); high capacity −0.025
(CI −0.035..−0.017, 17/18 wins, −31 %). Vs random: −57 %, −56 %, −44 %
(30/30, 45/45, 18/18 wins). The confirmation win held on each of the 9
boards individually. Classification: **POSITIVE** in all three runs.

Improvement per added infoset (discovery): `br_gain` 0.0039 chips/infoset
vs uniform 0.0031 and random 0.0015. Wall time: an 8-split adaptive run
costs 9 exact solves (~1 s here); uniform costs 1 solve. The oracle costs
~10x the adaptive run for a further 3–10 % reduction.

## What the negative signal says

Cumulative positive regret of the abstract solver (what a solver "sees"
during training) does not identify the cells whose *members* disagree: it
identifies cells with large, well-resolved regret mass. Error-guided
refinement needs a measure of heterogeneity *inside* the cell (per-hand
best-response gain, per-hand action disagreement, or even plain equity
spread), not the solver's own regret. This matters for the Hold'em
release: the solver's regret tables cannot be used directly as a refinement
signal, while collision dispersion (Phase I's `collision_dispersion`) is the
right kind of signal.

## Limitations (1I)

* single-street, perfect-recall domain; the Hold'em release abstraction is
  multi-street and imperfect-recall, where splitting a bucket changes reach
  distributions upstream;
* cells are contiguous in strength order and split at the median only;
  non-contiguous or feature-based splits were not tested;
* the error is in chips of a pot-10 subgame and must not be compared with
  full-game bb/100 figures;
* greedy one-split-per-round refinement; the oracle is itself one-step.

## Generalization (1J)

Not attempted on a multi-street Hold'em subgame in this sprint: the generic
tree solver takes minutes per exact solve on the Phase 56/57 flop/turn
subgames, so a capacity-matched multi-seed study was out of budget. The
release strategy was not modified.
