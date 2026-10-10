# Rollout response model (Phases 59-60)

How well does the Monte Carlo rollout pick actions, compared with exact
equilibria, and does a better opponent-response model help?

```
python experiments/rollout_response.py --iters 600 --sims 600 --range 10 \
    --models behavior,mdf_range,mdf_calibrated
```

-> `results/validation/rollout_response_v1.json` (diagnosis) and
`results/validation/rollout_response_v2.json` (adoption decision).

## Setup

96 exact river subgames (52-card deck): 6 board types (dry high, paired,
four-straight, four-flush, monotone, low connected) x 2 range shapes
(polarized vs bluff-catchers, merged vs merged) x bet sizes 0.5 / 1 / 2 pot
x SPR 1 / 2.5 / 5 (size <= SPR), one size + all-in, at most one raise, ranges
of 10 hands. Each is solved with CFR+ (600 iterations; exploitability
< 0.003 BB on the logged games). Games are split deterministically into
calibration and held-out halves (480 hero decisions each); adoption uses
held-out results only.

For every out-of-position hand the rollout (600 samples) picks check / bet /
all-in; this is compared with the exact action values Q of the equilibrium:
agreement (rollout pick = argmax Q), EV regret (max Q - Q(pick)), EV bias
(rollout EV - Q of the pick), bet-size, bluff and value-bet bias.

## Diagnosis (current default, `behavior`)

| | calibration | held-out |
| --- | --- | --- |
| agreement | 0.63 | 0.61 |
| mean / max regret (BB, pot 10) | 0.18 / 4.64 | 0.24 / 4.43 |
| EV bias (BB) | +0.70 | +0.78 |
| bluff-bet frequency, rollout / equilibrium | 0.42 / 0.33 | 0.41 / 0.34 |
| value-bet frequency, rollout / equilibrium | 0.78 / 0.59 | 0.72 / 0.59 |

The rollout is **optimistic** (its EV for the chosen action is ~0.7-0.8 BB
higher than the equilibrium value) and **bets too often**, both as a bluff
and for value. Regret grows with SPR (0.05 BB at SPR 1, 0.29 BB at SPR 5)
and is largest with polarized vs bluff-catcher ranges (0.30 vs 0.12 BB
merged) and on paired boards (0.33 BB). Responder side: the behaviour model
folds 12 points too little to half-pot bets and about right to larger ones
(mean equilibrium fold 0.61 vs 0.59).

Breakdowns by range strength, nut advantage and required call equity are
represented by the range-shape, board and size splits above; no separate
continuous breakdown was computed.

## Response model v2 candidates (Phase 60)

| held-out | behavior (default) | mdf_range | mdf_calibrated |
| --- | --- | --- | --- |
| agreement | **0.61** | 0.56 | 0.53 |
| mean regret (BB) | **0.24** | 0.26 | 0.30 |
| EV bias (BB) | **+0.78** | +1.08 | +1.16 |
| bluff-bet frequency (eq. 0.34) | **0.41** | 0.49 | 0.66 |

The calibrated defend share fitted on the calibration games is
a + b/(1+s) with a = -0.01, b = 1.00: the equilibrium defends almost exactly
the minimum-defence frequency, so `mdf_calibrated` ~= `mdf_range`. Their
*responder* fold rates are closer to equilibrium, but because those
responders never raise, bets look safer to the hero and the rollout
over-bets more.

**Decision: not adopted.** The default stays `rollout_response="behavior"`;
the MDF models remain opt-in. A 4-game pilot had suggested the opposite
(0.75 vs 0.31 agreement); the 96-game run with a held-out split does not
reproduce it — a reminder that small pilots are not evidence.

Scope: river only (depth is irrelevant on the river; see the depth study in
`docs/decision_engine.md`), heads-up, small ranges. These are exact
equilibria of small subgames, not of real Hold'em.
