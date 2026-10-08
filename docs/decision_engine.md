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

None of B, C or the fallback is labelled GTO. Even method A is an
approximate equilibrium of an *abstracted heads-up* game; the report says so.

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
