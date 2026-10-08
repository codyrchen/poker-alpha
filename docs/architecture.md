# Architecture

PokerAlpha has two layers that share code but not assumptions:

* the **research core** (Kuhn, Leduc, CFR/CFR+/MCCFR, exact evaluation,
  opponent-identification experiments) — unchanged and still pinned by
  reproducibility digests (`tests/test_reproducibility.py`);
* the **Hold'em decision-support platform** built around it.

```
 Screen / hand history / simulator / manual entry      (input adapters)
                         ↓
               ObservedTableState                       (poker_alpha.holdem.observed)
                         ↓  validate()
            Range + opponent model                      (poker.ranges, opponent.*)
                         ↓
        Solver lookup / equity / EV rollouts            (solvers, poker.multiway, decision.rollout)
                         ↓
                  DecisionReport                        (decision.report)
                         ↓
         UI / replay / session analysis                 (ui, replay, session)
```

**Dependency rule.** Solver and poker logic never import the observer, the
UI, or any site-specific code. The observer (`poker_alpha.observer`) depends
on the core only through `ObservedTableState`; nothing depends on it.
Optional dependency groups keep it that way: `[vision]` (Pillow, mss),
`[ocr]` (pytesseract), `[ui]` (Streamlit).

## Packages

| package | role |
| --- | --- |
| `games/` | two-player zero-sum extensive-form games for CFR: Kuhn, Leduc, abstracted HU Hold'em (`HoldemGame`, now with a pluggable information-state encoder) |
| `solvers/` | CFR, CFR+, MCCFR (sampled chance), exact evaluation, canonical strategy digests, versioned `.npz` checkpoints, Hold'em run diagnostics |
| `abstraction/` | information-state encoders (raw / toy / bucketed), 169 preflop classes, board and hand features, betting abstraction |
| `holdem/` | 2–9 seat no-limit rules engine (integer chips, side pots), positions, `ObservedTableState`, input adapters |
| `poker/` | cards, evaluator, heads-up equity, combo-level `WeightedRange`, multiway equity |
| `opponent/` | archetypes and Bayesian beliefs (research), behaviour models, range priors and updates, player statistics |
| `decision/` | `recommend_action` → `DecisionReport`, solver lookup, Monte Carlo action rollouts |
| `history/` | canonical events, `pokeralpha.hand/v1` JSON, replay through the rules engine |
| `observer/` | screen sources, calibration, OCR/card recognition, PokerNow-style adapter, smoothing and fusion |
| `session/` | SQLite session store, post-session analysis |
| `ui/` | Streamlit decision-support app (display only) |

## Versioned formats

| format | identifier |
| --- | --- |
| strategy digest | `PASD` v1 (binary, see `solvers/digest.py`) |
| solver checkpoint | `.npz`, `format_version` 1, pickle-free |
| encoder signatures | `RawHoldemEncoder:v1`, `ToyHoldemEncoder:v1`, `HoldemBucketEncoder:v1:equity=..` |
| action abstraction | `ActionAbstraction:v1:bets=..:raises=..:allin=..` |
| observed state | `pokeralpha.observed/v1` |
| hand history | `pokeralpha.hand/v1` |
| range priors | `pokeralpha.preflop_ranges/v1` |
| calibration | `pokeralpha.calibration/v1` |
| session database | SQLite, `meta.schema_version` = 1 |

## What is (and is not) theoretically grounded

| component | status |
| --- | --- |
| CFR/CFR+ on Kuhn/Leduc | converges to Nash in two-player zero-sum games; exploitability computed exactly |
| MCCFR on abstracted HU Hold'em | converges (in expectation) to an equilibrium **of the abstract game**; exploitability in real Hold'em is unknown and not reported |
| bucketed abstraction | perfect recall within the abstraction (bucket history + exact actions); lossy, not optimal |
| toy encoder | imperfect recall; plumbing demo only |
| multiplayer (3–9) | **no** equilibrium claim; CFR guarantees do not extend to multiplayer general-sum play |
| range estimates | Bayesian *beliefs* under heuristic priors and behaviour models, not known hands |
| rollouts | approximate local EVs under a stated response model and check-down assumption |
| observer | synthetic-fixture validated only; OCR adds state uncertainty |

## Permitted use

The observer reads pixels. Nothing in PokerAlpha clicks, types, controls a
browser or submits actions. Real-time assistance is for private, play-money
or test games where it is allowed; elsewhere use the same pipeline after the
session (hand-history replay, screenshot import, `python -m
poker_alpha.session`).
