"""Abstracted heads-up no-limit Texas Hold'em.

A research-oriented, deliberately restricted HUNL: correct core rules with an
action abstraction that keeps the betting tree tractable for CFR-style solving
and simulated matches. Obscure rules (min-raise technicalities, side pots —
irrelevant heads-up) are simplified per the project brief.

Setup
-----
* 2 players, 100 BB starting stacks, blinds 0.5 / 1 BB.
* Player 0 is the button/small blind: acts **first preflop**, **second** on
  every postflop street. Player 1 posts the big blind.
* Streets: preflop, flop (3 cards), turn, river.

Action abstraction (single tokens in the per-street history):

* ``f``    — fold (only when facing chips to call)
* ``c``    — check, or call the outstanding amount
* ``b50``  — bet/raise adding 0.5 × pot-after-call
* ``b100`` — bet/raise adding 1.0 × pot-after-call
* ``b200`` — bet/raise adding 2.0 × pot-after-call
* ``a``    — all-in (the whole remaining stack)

Only actions that are legal in the current state are offered: bets are dropped
when they exceed the stack or don't exceed the current outstanding amount, and
a per-street raise cap bounds the tree. Chip amounts are in BB (floats).

The class implements the :class:`~poker_alpha.games.base.Game` interface so
subgame solvers can run on it, and adds :meth:`deal`/:meth:`sample_chance` for
Monte Carlo use — enumerating every Hold'em deal is exactly what MCCFR exists
to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from ..poker.cards import NUM_CARDS
from ..poker.evaluator import evaluate_best
from .base import Game

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
_BOARD_SIZE = {PREFLOP: 0, FLOP: 3, TURN: 4, RIVER: 5}

SMALL_BLIND = 0.5
BIG_BLIND = 1.0
STARTING_STACK = 100.0

_BET_FRACTIONS = {"b50": 0.5, "b100": 1.0, "b200": 2.0}
_RAISE_CAP_PER_STREET = 3  # bets/raises per street, keeps the tree bounded


@dataclass(frozen=True)
class HoldemState:
    """One node of the abstracted HUNL game.

    ``holes`` is ``None`` before the deal (root chance node); ``board`` grows
    at street-transition chance nodes. ``contrib`` are total chips committed
    by each player across all streets; ``streets`` is a tuple of per-street
    action strings, e.g. ``("cc", "b100c", "")``.
    """

    holes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    board: Tuple[int, ...]
    streets: Tuple[str, ...]
    contrib: Tuple[float, float]
    folded: int = -1  # player index who folded, or -1
    all_in: bool = False

    @property
    def street(self) -> int:
        return len(self.streets) - 1

    @property
    def pot(self) -> float:
        return self.contrib[0] + self.contrib[1]


def _tokens(street_actions: str) -> List[str]:
    """Split a street's action string into action tokens."""
    out: List[str] = []
    i = 0
    while i < len(street_actions):
        ch = street_actions[i]
        if ch in ("f", "c", "a"):
            out.append(ch)
            i += 1
        elif ch == "b":
            j = i + 1
            while j < len(street_actions) and street_actions[j].isdigit():
                j += 1
            out.append(street_actions[i:j])
            i = j
        else:
            raise ValueError(f"bad action char {ch!r} in {street_actions!r}")
    return out


class HoldemGame(Game):
    """Abstracted heads-up NL Hold'em (see module docstring)."""

    def __init__(self,
                 starting_stack: float = STARTING_STACK,
                 bet_fractions: Optional[dict] = None,
                 raise_cap: int = _RAISE_CAP_PER_STREET) -> None:
        self.starting_stack = float(starting_stack)
        self.bet_fractions = dict(bet_fractions or _BET_FRACTIONS)
        self.raise_cap = raise_cap

    # -- construction / chance ------------------------------------------

    def root(self) -> HoldemState:
        return HoldemState(holes=None, board=(), streets=("",),
                           contrib=(SMALL_BLIND, BIG_BLIND))

    def is_chance(self, state: HoldemState) -> bool:
        if state.holes is None:
            return True
        if state.folded != -1:
            return False
        # Board deal pending: street betting closed but board short for the
        # street we're entering; also runouts after an all-in call.
        if self._betting_closed(state) and len(state.board) < 5:
            return True
        return False

    def sample_chance(self, state: HoldemState,
                      rng: np.random.Generator) -> HoldemState:
        """Sample one chance outcome (deal) without enumerating them all."""
        if state.holes is None:
            cards = rng.choice(NUM_CARDS, size=4, replace=False)
            holes = ((int(cards[0]), int(cards[1])),
                     (int(cards[2]), int(cards[3])))
            return replace(state, holes=holes)
        dead = set(state.board) | {c for h in state.holes for c in h}
        live = [c for c in range(NUM_CARDS) if c not in dead]
        need = _BOARD_SIZE[state.street + 1] - len(state.board)
        picked = rng.choice(len(live), size=need, replace=False)
        new_board = state.board + tuple(int(live[i]) for i in picked)
        new_streets = state.streets + ("",)
        return replace(state, board=new_board, streets=new_streets)

    def chance_outcomes(self, state: HoldemState):
        raise NotImplementedError(
            "full Hold'em chance enumeration is intentionally unsupported; "
            "use sample_chance (MCCFR / simulation) or solve fixed-board "
            "subgames"
        )

    def deal(self, rng: np.random.Generator) -> HoldemState:
        """Deal a fresh hand: sample the root chance node."""
        return self.sample_chance(self.root(), rng)

    # -- betting mechanics ----------------------------------------------

    def _replay(self, state: HoldemState) -> Tuple[List[float], List[float], int, int]:
        """Replay the whole betting history from the blinds forward.

        Returns ``(street_paid, total, to_act, n_raises)`` where ``total`` is
        each player's cumulative contribution (blinds included), ``street_paid``
        their contribution on the current street, and ``n_raises`` the raise
        count on the current street. Deriving everything from the action
        history — never from ``state.contrib`` mid-replay — is what keeps
        stack and pot arithmetic consistent (``contrib`` is a cache updated in
        ``next_state``, and using it *during* a replay would double-count the
        very actions being replayed).
        """
        total = [SMALL_BLIND, BIG_BLIND]
        street_paid = [SMALL_BLIND, BIG_BLIND]
        to_act = 0
        n_raises = 0
        for street_idx, street_actions in enumerate(state.streets):
            if street_idx > 0:
                street_paid = [0.0, 0.0]
                to_act = 1  # big blind first postflop
            else:
                street_paid = [SMALL_BLIND, BIG_BLIND]
                to_act = 0  # button first preflop
            n_raises = 0
            for tok in _tokens(street_actions):
                me, opp = to_act, 1 - to_act
                owe = street_paid[opp] - street_paid[me]
                stack = self.starting_stack - total[me]
                if tok == "c":
                    add = min(owe, stack)
                elif tok == "a":
                    add = stack
                    n_raises += 1
                elif tok == "f":
                    add = 0.0
                else:
                    pot_now = total[0] + total[1]
                    add = owe + self.bet_fractions[tok] * (pot_now + owe)
                    n_raises += 1
                street_paid[me] += add
                total[me] += add
                to_act = 1 - to_act
        return street_paid, total, to_act, n_raises

    def _betting_closed(self, state: HoldemState) -> bool:
        """Has the current street's betting concluded (no fold)?"""
        if state.all_in:
            # After a shove, betting is closed once the shove is matched
            # (equal totals); remaining streets are pure runout. An unmatched
            # shove still awaits the opponent's fold/call.
            return abs(state.contrib[0] - state.contrib[1]) < 1e-9
        actions = _tokens(state.streets[-1])
        if not actions:
            return False
        last = actions[-1]
        if last == "f":
            return True
        # A 'c' closes the street only as a response — a call of a bet, a
        # check-back, or the big blind checking their option. A first-action
        # 'c' (postflop check-open, or the preflop limp) leaves the other
        # player still to act. Both cases reduce to: len(actions) >= 2.
        return last == "c" and len(actions) >= 2

    # -- Game interface --------------------------------------------------

    def is_terminal(self, state: HoldemState) -> bool:
        if state.holes is None:
            return False
        if state.folded != -1:
            return True
        return len(state.board) == 5 and self._betting_closed(state)

    def utility(self, state: HoldemState) -> float:
        if state.folded == 0:
            return -state.contrib[0]
        if state.folded == 1:
            return state.contrib[1]
        h0 = evaluate_best(list(state.holes[0]) + list(state.board))
        h1 = evaluate_best(list(state.holes[1]) + list(state.board))
        if h0 > h1:
            return state.contrib[1]
        if h1 > h0:
            return -state.contrib[0]
        return 0.0

    def current_player(self, state: HoldemState) -> int:
        _, _, to_act, _ = self._replay(state)
        return to_act

    def infoset_key(self, state: HoldemState) -> str:
        player = self.current_player(state)
        hole = ",".join(str(c) for c in sorted(state.holes[player]))
        board = ",".join(str(c) for c in state.board)
        history = "/".join(state.streets)
        return f"{player}|{hole}|{board}|{history}"

    def legal_actions(self, state: HoldemState) -> List[str]:
        street_paid, total, to_act, n_raises = self._replay(state)
        me, opp = to_act, 1 - to_act
        owe = street_paid[opp] - street_paid[me]
        my_stack = self.starting_stack - total[me]
        opp_stack = self.starting_stack - total[opp]

        legal: List[str] = []
        if owe > 1e-9:
            legal.append("f")
        legal.append("c")
        # Raising: allowed under the cap, if we have chips beyond the call and
        # the opponent can still respond (no raising a player who is all-in).
        if n_raises < self.raise_cap and my_stack > owe + 1e-9 \
                and opp_stack > 1e-9:
            pot_now = total[0] + total[1]
            for name, frac in self.bet_fractions.items():
                add = owe + frac * (pot_now + owe)
                if add < my_stack - 1e-9:  # strictly less: 'a' covers the top
                    legal.append(name)
            legal.append("a")
        return legal

    def next_state(self, state: HoldemState, action: str) -> HoldemState:
        street_paid, total, to_act, _ = self._replay(state)
        me, opp = to_act, 1 - to_act
        owe = street_paid[opp] - street_paid[me]
        my_stack = self.starting_stack - total[me]
        folded = state.folded
        all_in = state.all_in
        if action == "f":
            folded = me
            delta = 0.0
        elif action == "c":
            delta = min(owe, my_stack)
        elif action == "a":
            delta = my_stack
            all_in = True
        else:
            pot_now = total[0] + total[1]
            delta = owe + self.bet_fractions[action] * (pot_now + owe)

        contrib = list(state.contrib)
        contrib[me] += delta
        streets = state.streets[:-1] + (state.streets[-1] + action,)
        return HoldemState(
            holes=state.holes,
            board=state.board,
            streets=streets,
            contrib=(contrib[0], contrib[1]),
            folded=folded,
            all_in=all_in,
        )
