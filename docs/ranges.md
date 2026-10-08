# Ranges and opponent modelling

**A range is a belief, not a hand.** Every range in PokerAlpha is a
probability distribution produced by priors and models that can be wrong;
reports always carry its entropy / effective combo count.

## `WeightedRange` (`poker/ranges.py`)

Non-negative weights over the 1,326 two-card combos (NumPy vector in
`itertools.combinations` order). Operations return new ranges: normalize,
`remove_cards` / `condition_on_board` / `condition_on_known_cards` (exact
vector masks), `class_mass`, `class_distribution`, `entropy_bits`,
`effective_combos`, seeded `sample` that never returns a blocked combo,
Bayesian `update(likelihood)` (raises if the observation is impossible under
the range). Range notation: `AA, AKs, AKo, AK, TT+, A2s+, KTo+, 22-66,
A5s-A2s, AA:0.5, random`.

## Priors (`opponent/ranges.py`, `data/preflop_ranges_v1.json`)

Versioned JSON keyed by stack bucket, preflop line (`open`, `limp`,
`call_open`, `3bet`, `call_3bet`, `4bet`, `call_4bet`, `unopened`,
`check_option`) and position, with aliases (UTG+1 → UTG …) and defaults.
The bundled table is a hand-written, illustrative approximation of typical
online cash ranges — **not** solver output. Pass your own file to
`RangePriors.load(path)`. `classify_preflop_line(actions, seat)` names a
seat's line from observed actions.

## Updates

`update_range_for_action(prior, action, facing_bet, board, dead, model,
size_pot_fraction)` applies `P(hand | action) ∝ P(action | hand, state) ·
P(hand)` combo-wise. Likelihoods come from:

* `BehaviorModel` — logistic thresholds on hand *strength* (preflop: class
  equity vs random, rescaled; postflop: share of live combos beaten on the
  current board), with archetypes `regular`, `nit`, `calling_station`,
  `maniac`. Facing a bet of `f × pot`, the fold and raise/call-off thresholds
  move with the required equity `f / (1 + 2f) − 0.25` (bounded), and raise
  bluffs vanish as it approaches 0.5. All parameters are illustrative
  defaults, not fitted to data.
* `StrategyLikelihood` — an explicit per-combo policy (e.g. a solver
  lookup) with a behavioural fallback.

Strength percentiles measure *current* made-hand strength only; draws are
not credited.

## Player statistics (`opponent/statistics.py`)

VPIP, PFR, 3-bet, fold to 3-bet, 4-bet, limp, cold call, flop c-bet, fold
to c-bet, turn and river barrels, check-raise, fold to raise, aggression
frequency, WTSD and W$SD — overall, by position and by street. Each is a
Beta posterior with geometric recency decay (the same forgetting scheme as
the research `ArchetypeBelief`), reporting raw counts, posterior mean and a
credible interval: 3/4 and 300/400 have the same raw rate and very
different intervals. Hand summaries come from replay (`ReplayResult.summary`)
or the session store.

## Multiway equity (`poker/multiway.py`)

Hero vs 1–8 ranges. Opponent hands are drawn from the exact joint
distribution `Π w_i(h_i) · 1[no collisions]` by rejection sampling (default)
or self-normalized importance sampling. Sampling opponents one after another
from blocker-adjusted ranges is biased; a test shows a case where it gives
0.25 instead of the exact 1/3. Optional `contributions` make the share
side-pot aware. Throughput on this machine after the Phase 22 optimization:
see `results/data/holdem_benchmark_optimized.csv`.
