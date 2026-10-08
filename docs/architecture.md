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

All four input sources go through `poker_alpha.pipeline` (`observe_*` ->
`analyze`), the only entry into the decision engine.

**Dependency rule.** Solver and poker logic never import the observer, the
UI, or any site-specific code. The observer (`poker_alpha.observer`) depends
on the core only through `ObservedTableState`; nothing depends on it.
Optional dependency groups keep it that way: `[vision]` (Pillow, mss),
`[ocr]` (pytesseract), `[ui]` (Streamlit).

## Layers and what each one claims

| layer | what it is | claim |
| --- | --- | --- |
| Research core | Kuhn / Leduc CFR, CFR+, MCCFR, exact exploitability, adaptive opponent research | equilibria of those small games, measured exactly |
| Hold'em engine | 2-9 player rules and simulation (`holdem/`) | correct rules, chip conservation, side pots (tested) |
| HU solver | approximate abstract MCCFR on heads-up 100 BB (`games/holdem.py`, `solver_config.py`) | abstract strategy only; imperfect recall; exploitability of the full abstraction unknown; gated before use |
| Multiway | range/EV approximation (equity, rollouts) | no Nash solving for 3+ players |
| Observer | read-only visual state extraction (`observer/`) | synthetic-fixture validated; real client BLOCKED on fixtures |
| Decision engine | solver -> rollout -> heuristic cascade with uncertainty (`decision/`, `pipeline.py`) | every number carries source and uncertainty |

## Packages

| package | role |
| --- | --- |
| `games/` | two-player zero-sum extensive-form games for CFR: Kuhn, Leduc, abstracted HU Hold'em (`HoldemGame`, now with a pluggable information-state encoder) |
| `solvers/` | CFR, CFR+, MCCFR (sampled chance), exact evaluation, canonical strategy digests, versioned `.npz` checkpoints, Hold'em run diagnostics |
| `abstraction/` | information-state encoders (raw / toy / bucketed / transition / compact), 169 preflop classes, board and hand features, cheap cached card features, betting-history abstraction |
| `solver_config.py` | locked `HoldemSolverConfig` v1 (`PRIMARY_CONFIG`: compact encoder, imperfect recall) |
| `pipeline.py` | one normalized path: manual / simulation / hand history / screenshot -> `DecisionReport` |
| `decision/solver_gate.py` | solver-confidence gate (accept / low confidence / reject with reasons) |
| `games/reduced_holdem.py` | exact Hold'em-shaped validation games (reduced preflop, fixed-board river) |
| `validation/` | abstraction audits, canonical spots and the canonical matrix, duplicate cross-play |
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
| solver checkpoint | `.npz`, `format_version` 2 (adds `solver_config`; version 1 still readable), pickle-free |
| solver config | `HoldemSolverConfig:v1:<hash>` (locked reference), `HoldemSolverConfig:v2:<hash>` (release: legal sizing, river percentiles, averaging recorded) |
| solver confidence table | `pokeralpha.solver_confidence/v1` (`.npz`, per-key visits / movement / seed disagreement / collision) |
| screenshot annotation | `pokeralpha.screenshot_annotation/v1` (validated by `validate_annotation`) |
| strategy artifact | `pokeralpha.strategy_artifact/v1` (`.npz`, average strategy + visits) |
| encoder signatures | `RawHoldemEncoder:v1`, `ToyHoldemEncoder:v1`, `HoldemBucketEncoder:v1:equity=..`, `TransitionHoldemEncoder:v1:..`, `CompactHoldemEncoder:v1:..:recall=imperfect` |
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
| MCCFR on abstracted HU Hold'em, perfect-recall encoders (raw, bucket, transition) | converges (in expectation) to an equilibrium **of the abstract game**; exploitability in real Hold'em is unknown and not reported |
| MCCFR on the locked compact encoder (primary) | **imperfect recall: no standard CFR guarantee**; produces an abstract heuristic strategy whose quality is measured only by proxies (visits, seed agreement, sanity checks, cross-play) |
| bucketed abstraction | perfect recall within the abstraction (bucket history + exact actions); lossy, not optimal; too fine to be revisited (Phase 25/26) |
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
