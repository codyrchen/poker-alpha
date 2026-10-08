# Decision engine (`poker_alpha.decision`)

```python
from poker_alpha.decision import DecisionConfig, recommend_action
report = recommend_action(observed_state, config=DecisionConfig(
    equity_simulations=3000, rollout_simulations=2000, seed=0))
print(report.format())
```

## Method hierarchy and provenance

| method | when | `source` label |
| --- | --- | --- |
| A. solver | heads-up spot that maps into a trained abstract strategy (same stack depth and blinds, translatable history, enough visits) | `solver` (exact mapping) / `interpolated abstraction` (off-tree sizes translated) |
| B. range rollout | heads-up outside the abstraction | `Monte Carlo rollout` |
| C. multiway estimate | 3+ players in the hand | `Monte Carlo rollout` (with a multiway warning) |
| fallback | rollouts disabled | `heuristic fallback` (pot odds, check-down EVs) |

None of B, C or the fallback is labelled GTO. Method A is not an
equilibrium either: the locked solver config (`HoldemSolverConfig` v1,
`docs/abstraction.md`) uses an **imperfect-recall** abstraction, so its
average strategy is an abstract MCCFR strategy with no equilibrium
guarantee, even of the abstract game. Reports say so in `mix_meaning`,
`uncertainty["abstraction"]` and a warning.

### Source priority and rejection reasons (Phase 31)

Priority is solver -> Monte Carlo rollout -> heuristic. Every report carries
`details["source_cascade"]`: one entry per source with its status (`used`,
`rejected`, `not configured`, `not run`, `not needed`, `used for EVs only`)
and, for a rejected solver, a code from `decision.strategy.REJECTION_CODES`:

| code | meaning |
| --- | --- |
| `config_mismatch` | the strategy file was trained under a different solver config / encoder / game signature |
| `incompatible_checkpoint` | file missing, unreadable or of an unsupported format |
| `out_of_abstraction` | the spot cannot be mapped (not heads-up, other blinds or stack depth, untranslatable history) |
| `unvisited` | the abstract information set was never visited in training |
| `insufficient_visits` | visited fewer than `min_visits` times (default 20) |

When the solver is used and rollouts are enabled, a consistency check
compares the solver's most frequent action with the highest rollout EV; if
rollouts favour another action by more than two paired standard errors, the
report warns ("solver and rollouts disagree"), drops confidence to low and
adds a `consistency check: conflict` cascade entry. With the 300k strategy
this fires, for example, on BTN AQo at 100 BB (solver limps, rollouts prefer
a raise) — such spots should be treated as uncertain.

`pipeline.load_solver(path)` (artifact or checkpoint) returns either a
`SolverStrategyProvider` or a coded `LookupMiss`, which `pipeline.analyze`
reports in the cascade instead of failing.

### Uncertainty by source

`report.uncertainty` keeps five sources separate instead of blending them:
`observation` (given state vs screen recognition and its confidence),
`range_estimation` (effective combos and entropy of each opponent belief),
`sampling` (equity SE, rollout EV SE), `abstraction` (solver infoset, visits,
exact vs translated sizes, strategy description) and `response_model`
(rollout behaviour models / check-down assumption / none).

## One path for every input (`poker_alpha.pipeline`)

```python
from poker_alpha.pipeline import observe_manual, analyze, load_solver
report = analyze(observe_manual(state_dict), solver=load_solver("results/strategy/holdem_v1_seed0.npz"))
```

`observe_manual`, `observe_simulation`, `observe_hand_history` and
`observe_screenshot` all produce an `Observation` (an `ObservedTableState`
plus its source and, for screenshots, the observer confidence); `analyze`
is the single entry into `recommend_action`. Demo:
`python -m poker_alpha.platform_demo`.

## `DecisionReport`

State summary, hero equity ± SE (vs the estimated ranges), pot odds, SPR,
effective stack, opponent range summaries (position, preflop line, live and
effective combos, entropy, top classes, behaviour model), candidate actions
(label, concrete amount, frequency, EV in BB, SE, samples, source, note),
recommendation, recommended mix **and what the mix means**, confidence
(low/medium/high) and warnings (validation issues, solver refusals,
multiway, high-SPR all-ins, observer confidence).

For rollouts the "frequency" column is the paired-bootstrap probability that
the action has the highest EV — estimation uncertainty, not a mixed
strategy. Confidence is `high` only when the best action beats the
runner-up by more than 3 paired standard errors heads-up against a
reasonably narrow range.

## Rollouts (`decision/rollout.py`)

Per sample, shared by all candidates (common random numbers): joint
opponent hands from their ranges (exact rejection sampling), the runout,
and uniforms for every response. Each candidate is replayed: hero acts;
opponents respond fold/call/raise (at most one raise, to
`raise_multiplier ×` the bet) via their `BehaviorModel`; after a check,
opponents behind may bet `opponent_bet_fraction` of the pot; the hero
continues per `hero_model`; the hand is then checked down and settled with
the engine's side-pot code.

Known biases, stated rather than hidden:

* **check-down after one response** — later-street value of smaller bets is
  not credited, which favours big bets and all-ins at high SPR (reports
  warn);
* behaviour models are heuristic and use current strength, not draws;
* money from earlier streets is one pot layer for everyone remaining.

Validated against closed forms: fold EV is exactly 0; when the opponent is
already all-in the call EV matches `share × (pot + call) − call` computed by
exact enumeration; paired differences have lower variance than independent
ones; obvious folds, free checks, forced all-in calls, nut hands and pot-odds
boundaries give the expected answers (`tests/test_rollout.py`,
`tests/test_validation.py`).

### Decision-quality invariants (Phase 25)

`tests/test_decision_invariants.py` pins opponent responses with a
fixed-probability model so rollout EVs have closed forms, and asserts:
the nuts never folds; a zero-equity call costs exactly the call; fold vs
call flips exactly at the pot-odds threshold; an always-fold opponent makes
every bet worth exactly the pot; an always-call river bet is worth
`pot + X` (win) or `−X` (lose); with equity below one half and constant
responses, EV strictly decreases with bet size (paired differences equal
`(2s − 1)ΔX` exactly) and the engine recommends checking; multiway pot
shares are not double-counted; folded players never win; and side-pot
eligibility is respected (a short all-in nut hand wins only the main pot).

The suite found one real bug: equity and rollout settlement used integer
pot units, so the odd-chip rule biased split pots (a 2-way tie in a 3-way
pot scored 2/3 vs 1/3). Analysis now scales pot units by 2520 = lcm(1..9) so
ties split exactly; the rules engine keeps the real odd-chip rule.

## Replay and post-session use

`python -m poker_alpha.replay hand.json --recommend --rollouts 1000` prints a
report for every hero decision next to the action actually taken.
`python -m poker_alpha.session import hand.json --db s.sqlite` stores hands,
snapshots, reports and EV gaps; `analyze` lists the largest EV deviations
(with SE), uncertain decisions, range narrowing, player tendencies with
credible intervals and hands worth reviewing.
