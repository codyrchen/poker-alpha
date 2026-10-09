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
| `CONFIG_MISMATCH` | the strategy file was trained under a different solver config / encoder / game signature |
| `INCOMPATIBLE_CHECKPOINT` | file missing, unreadable or of an unsupported format |
| `OUTSIDE_ABSTRACTION` | the spot cannot be mapped (not heads-up, other blinds or stack depth, untranslatable history) |
| `UNSEEN_STATE` | the abstract information set was never visited in training |
| `LOW_VISIT_COUNT` | visited fewer than `min_visits` times (default 20) |

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

### Solver-confidence gate (Phase 37)

A solver key that exists is not trusted by default. `decision/solver_gate.py`
reads the strategy's confidence table (per key: visits, movement between
checkpoints, seed disagreement, collision dispersion, audit flags) and
returns `SOLVER_ACCEPT`, `SOLVER_LOW_CONFIDENCE` (used, confidence low,
warning) or `SOLVER_REJECT` (fall back to rollout, then heuristic). Reasons
are machine-readable: `LOW_VISIT_COUNT`, `HIGH_SEED_DISAGREEMENT`,
`UNSTABLE_ACROSS_CHECKPOINTS`, `HIGH_COLLISION_DISPERSION`,
`CONFIG_MISMATCH`, `INCOMPATIBLE_CHECKPOINT`, `UNSEEN_STATE`,
`OUTSIDE_ABSTRACTION`, `KNOWN_PATHOLOGICAL_BUCKET`, `NO_STABILITY_DATA`.
Thresholds are calibrated against true strategy error in exact games
(`docs/solver_validation.md`, Phase 37). Every report has
`details["solver"]` = {used, confidence, reasons, visits, seed_disagreement,
movement, collision}; the UI prints "Solver strategy not used: <reasons> ·
fallback: <method>" when it is rejected.

The lookup also never returns a bet or raise below the NLHE minimum (the v1
abstraction contains such sizes); their probability mass is removed,
renormalized and reported as `illegal_size_mass_removed`.

Known weakness found by exact games: the default opponent behaviour model
folds too much to large bets, so rollouts recommend river overbets where
the exact equilibrium checks (16 of 36 spots); rollout EVs themselves match
closed forms (`results/validation/rollout_validation_v1.json`).

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
report = analyze(observe_manual(state_dict), solver=load_solver("results/strategy/holdem_v2_seed0.npz"))
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

## Fault handling (Phase 68)

`recommend_action` never invents a recommendation from a state it cannot
trust. Tested in `tests/test_decision_faults.py`.

**Refusals** (`recommended = None`, `method = "none"`, codes in
`details["refusal_codes"]`):

| code | when |
| --- | --- |
| validation error codes (`duplicate_cards`, `pot_below_bets`, `pot_mismatch`, `board_street`, `dealer`, ...) | `validate()` reports an error |
| `HERO_CARDS_UNKNOWN` | hero cards missing |
| `NOT_HERO_TURN` | a known actor other than the hero |
| `HERO_STACK_UNKNOWN` | hero stack unreadable: sizes, all-in and SPR are undefined |
| `HERO_FOLDED`, `HERO_ALL_IN`, `NO_OPPONENTS` | nothing to decide |

Input errors that cannot even form a state (bad card string, unknown dealer
or hero seat) raise `ValueError` with the field named.

**Documented assumptions** (recommendation made, confidence capped at
`low`, codes in `details["assumptions"]`, warning shown):

* `ACTOR_UNKNOWN` — assumed to be the hero's turn;
* `OPPONENT_STACK_UNKNOWN` — the opponent is assumed to cover the hero.

**Component failures** (reported in the source cascade with a code):

| failure | code | fallback |
| --- | --- | --- |
| solver `lookup()` raises | `SOLVER_ERROR` | treated as a solver rejection; rollout / heuristic |
| missing / corrupt / wrong-config artifact | `INCOMPATIBLE_CHECKPOINT` / `CONFIG_MISMATCH` | same |
| equity sampler raises | `EQUITY_ERROR` | no equity; rollouts if they work |
| opponent range empty after card removal | `RANGE_EMPTY` | no equity, no rollouts |
| rollout raises (e.g. `RolloutError`) | `ROLLOUT_ERROR` | heuristic fallback; on the solver path the solver frequencies are kept without EVs |
| heuristic facing a bet without equity | `NO_EQUITY` | **no recommendation** |

Without an equity estimate the heuristic only ever picks *check* (never worse
than folding) and says so in a warning.

## Rollout options: depth and opponent response (Phases 59-61)

`DecisionConfig(rollout_depth=..., rollout_response=..., rollout_response_params=...)`
(passed to `rollout_action_evs`). Defaults are unchanged and pinned by the
golden tests.

* `rollout_depth="street"` (default, fast): the current street is resolved
  (one opponent response, one re-raise round) and the hand is checked down.
* `rollout_depth="showdown"` (heads-up only; multiway silently uses
  "street" and says so in the candidate note): later streets are also played
  — on each board one player may bet `opponent_bet_fraction` x pot and the
  other calls or folds by its behaviour model on that street's hand strength
  (no later-street raises).
* `rollout_response`: `"behavior"` (default: `BehaviorModel` on absolute
  hand strength), `"mdf_range"` (the responder defends the top
  1/(1+size) of its own range, never raises), `"mdf_calibrated"` (defends
  a + b/(1+size), parameters fitted in Phase 60). Adoption of a non-default
  response model is decided in `docs/rollout_model.md`.

Depth study (`python experiments/rollout_depth_study.py --sims 1000`,
`results/validation/rollout_depth.json`): 7 heads-up spots x {regular,
calling station}. The recommendation differs between depths in **6 of 14**;
the median largest EV shift is 1.9 BB; acting on the fast recommendation
costs up to **6.8 BB** if the showdown model is right (turn call vs fold
with middle pair). Both depths are heuristic policies; neither is an
equilibrium, so the study shows model dependence, not which is correct.
Latency: median 0.23 s (street) vs 3.0 s (showdown; 9-11 s preflop, where
three future streets are simulated). **Decision: the live default stays
"street"; "showdown" is offered for offline / post-hand analysis.**

## Monte Carlo error calibration (Phase 62)

`python experiments/mc_error_calibration.py --reps 100 --rollout-reps 40 --rollout-ref 40000`
-> `results/validation/mc_error_calibration.json`. For n = 100 / 300 / 1000 /
2000 / 5000 samples, repeated with independent seeds: bias, RMSE, SE ratio
(mean reported SE / empirical SD; 1 = honest) and coverage of the nominal
95% interval.

**Equity** (5 spots; river, turn and 3-way turn references exact by
enumeration, preflop/flop 2M-sample references; 100 reps): coverage
0.89-0.99 (mean 0.94), SE ratio 0.85-1.17 (mean 1.00), |bias| <= 0.006
(within noise). The reported equity SE is honest at every n.

**Rollout EVs** (3 spots, all candidates; references 40k-sample estimates;
40 reps): mean coverage 0.95, mean SE ratio 1.05, |bias| < 0.5 empirical
SD. One failure: the **all-in candidate at n = 100** (preflop): coverage
0.68 — rare, large outcomes make the normal-approximation SE too small.
At n >= 300 every candidate covers >= 0.875 (40 reps: 0.875 is within ~2
binomial SE of 0.95). The UI default is 1,000 samples; below 500 the report
now warns that error bars for all-in / large bets are too narrow.
