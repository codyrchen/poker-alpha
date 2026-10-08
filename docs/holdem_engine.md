# Hold'em rules engine (`poker_alpha.holdem`)

A deterministic, immutable state machine for no-limit Texas Hold'em with
2–9 seats. It is independent of the CFR games: the solver's `HoldemGame`
(heads-up, float BB, fixed bet tokens) is left untouched for reproducibility.

## State

`HoldemTableState` (frozen dataclass): seats (`SeatState`: stack,
committed this street / this hand, folded, all-in, optional hole cards),
dealer, `TableConfig` (small blind, big blind, ante), street, board, actor,
current bet, last full raise increment, per-seat reopening levels, the
ordered action history, and awards once complete. **Chips are integers**, so
conservation is exact; replay converts decimal amounts with an explicit
`chip_scale` and rejects amounts that are not exact multiples.

## Rules

* Antes are dead money; blinds are street bets. Short blinds/antes post
  all-in. The big blind's nominal amount is what others must call.
* Heads-up, the button posts the small blind, acts first preflop and last
  postflop. 3+ handed: UTG first preflop, first live seat left of the button
  postflop.
* Minimum bet = one big blind; minimum raise = last full raise increment.
* A short all-in raise does not reopen betting for players who already
  acted, unless several short raises *cumulatively* reach a full raise over
  the level that player last faced (TDA rule 47 style).
* Folding is not offered when checking is free; nobody may raise when every
  opponent is all-in.
* Pots: `build_pots` layers contributions by live players' levels (folded
  chips are dead but count toward the layers they reach; uncalled excess
  becomes a pot only its owner can win). `award_pots` splits ties with odd
  chips going clockwise from the button.

## API

```python
from poker_alpha.holdem import (Action, TableConfig, start_hand, apply_action,
                                legal_actions, cards_needed, deal_board,
                                deal_board_random, settle)

s = start_hand([200, 150, 80], dealer=0, config=TableConfig(1, 2, ante=0),
               hole_cards=[...])
while not s.is_complete:
    if s.actor is not None:
        s = apply_action(s, Action.raise_to(6))   # or call/check/fold/all_in
    elif cards_needed(s):
        s = deal_board(s, flop_cards)             # or deal_board_random(s, rng)
    else:
        s = settle(s)
```

Illegal requests raise `IllegalActionError`. `check_invariants(state, total)`
asserts conservation, no duplicate cards, a legal actor and award integrity.

## Tests

`tests/test_pots.py` (side pots in isolation), `tests/test_holdem_engine.py`
(heads-up, 3-way, 6-max, 9-max, short stacks, overbets, all-in calls,
multiway all-ins, one stack covering everyone, multiple side pots, tied main
and side pots, dead money, min-raise and reopening rules) and property tests
over thousands of random 2–9 seat hands (`tests/test_validation.py`).

## Relation to the solver game

The solver's heads-up `HoldemGame` (`games/holdem.py`) is a separate,
float-BB abstract game. Since Phase 28 it memoizes its betting replay, legal
actions and the abstract betting context by action history (bounded
tables), so its tree parameters (stack, bet menu, raise cap) must not be
mutated after construction; build a new game (or a new
`HoldemSolverConfig`) instead. Training output is bit-identical to the
unmemoized code (`tests/test_performance_equivalence.py`). The rules engine
in this document is not used for solving and was not changed.

Since Phase 35 the solver game also has two opt-in options used by
`HoldemSolverConfig` v2: `preflop_raise_multiples` (preflop raises become
"raise to m x current bet" tokens `x200`/`x250`/`x350`) and
`enforce_min_raise` (no bet below 1 BB, no raise increment below the last
full increment on the street; blinds count as a 1 BB bet). Both default
off, so v1 checkpoints and pinned digests are unchanged
(`tests/test_legal_sizing.py`).
