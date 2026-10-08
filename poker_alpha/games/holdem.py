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
from functools import lru_cache
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
_MEMO_LIMIT = 1 << 18  # entries per memo table before it is cleared
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


@lru_cache(maxsize=1 << 16)
def _tokens(street_actions: str) -> Tuple[str, ...]:
    """Split a street's action string into action tokens (cached; the
    result is an immutable tuple)."""
    out: List[str] = []
    i = 0
    while i < len(street_actions):
        ch = street_actions[i]
        if ch in ("f", "c", "a"):
            out.append(ch)
            i += 1
        elif ch in ("b", "x"):
            j = i + 1
            while j < len(street_actions) and street_actions[j].isdigit():
                j += 1
            out.append(street_actions[i:j])
            i = j
        else:
            raise ValueError(f"bad action char {ch!r} in {street_actions!r}")
    return tuple(out)


class HoldemGame(Game):
    """Abstracted heads-up NL Hold'em (see module docstring)."""

    def __init__(self,
                 starting_stack: float = STARTING_STACK,
                 bet_fractions: Optional[dict] = None,
                 raise_cap: int = _RAISE_CAP_PER_STREET,
                 encoder=None,
                 preflop_raise_multiples: Optional[dict] = None,
                 enforce_min_raise: bool = False) -> None:
        # Imported lazily: the abstraction package imports game-side helpers.
        from ..abstraction.holdem import RawHoldemEncoder

        self.starting_stack = float(starting_stack)
        self.bet_fractions = dict(bet_fractions or _BET_FRACTIONS)
        self.raise_cap = raise_cap
        # Optional legal-NLHE sizing (HoldemSolverConfig v2). Preflop raises
        # become "raise to multiple x current bet" tokens ``x<100*m>``;
        # with ``enforce_min_raise`` any bet/raise below the NLHE minimum
        # (1 BB bet, raise increment >= the last full increment on the
        # street, blinds counting as a 1 BB bet) is not offered. Both default
        # off, which keeps the v1 game (and its checkpoints) unchanged.
        self.preflop_raise_multiples = dict(preflop_raise_multiples or {})
        for tok in self.preflop_raise_multiples:
            if not (tok.startswith("x") and tok[1:].isdigit()):
                raise ValueError(f"preflop token {tok!r} must be 'x' + digits")
        self.enforce_min_raise = bool(enforce_min_raise)
        # Information-state encoder (see poker_alpha.abstraction.base). The
        # default reproduces the historical raw key byte for byte.
        self.encoder = encoder if encoder is not None else RawHoldemEncoder()
        # Phase 28 memo tables keyed by the action history (``streets``),
        # which together with the tree parameters above fully determines the
        # betting replay and the legal actions. Tree parameters must not be
        # mutated after construction. Bounded: cleared when full.
        self._replay_memo: dict = {}
        self._legal_memo: dict = {}

    def encoder_signature(self) -> str:
        return self.encoder.signature()

    def signature(self) -> str:
        """Versioned game signature covering every tree-shaping parameter."""
        def num(x: float) -> str:
            return format(float(x), ".12g")

        # Bet order is part of the signature: it fixes legal-action order.
        fracs = ",".join(f"{name}={num(frac)}"
                         for name, frac in self.bet_fractions.items())
        sig = (f"HoldemGame:v1:stack={num(self.starting_stack)}:"
               f"blinds={num(SMALL_BLIND)}/{num(BIG_BLIND)}:"
               f"bets={fracs}:raise_cap={int(self.raise_cap)}")
        if self.preflop_raise_multiples:
            sig += ":preflop=" + ",".join(f"{t}={num(m)}" for t, m in self.preflop_raise_multiples.items())
        if self.enforce_min_raise:
            sig += ":min_raise=nlhe"
        return sig

    # -- sizing ------------------------------------------------------------

    def raise_add(self, tok: str, street: int, owe: float, pot_now: float,
                  my_street_paid: float) -> float:
        """Chips added by bet/raise token ``tok`` (not 'a'/'c'/'f')."""
        if tok[0] == "x":
            level = my_street_paid + owe
            return self.preflop_raise_multiples[tok] * level - my_street_paid
        return owe + self.bet_fractions[tok] * (pot_now + owe)

    def raise_tokens(self, street: int):
        if street == 0 and self.preflop_raise_multiples:
            return self.preflop_raise_multiples
        return self.bet_fractions

    def size_label(self, tok: str, facing: bool, street: int) -> str:
        if tok[0] == "x":
            return f"raise_to_{self.preflop_raise_multiples[tok]:g}x"
        pct = int(round(self.bet_fractions[tok] * 100))
        return f"{'raise' if facing else 'bet'}{pct}"

    def _min_increment(self, streets: Tuple[str, ...]) -> float:
        """Minimum legal raise increment on the current street (NLHE)."""
        memo = self.__dict__.setdefault("_inc_memo", {})
        hit = memo.get(streets)
        if hit is not None:
            return hit
        total = [SMALL_BLIND, BIG_BLIND]
        inc = BIG_BLIND
        for si, sa in enumerate(streets):
            paid = [SMALL_BLIND, BIG_BLIND] if si == 0 else [0.0, 0.0]
            to_act = 0 if si == 0 else 1
            inc = BIG_BLIND
            for tok in _tokens(sa):
                me, opp = to_act, 1 - to_act
                owe = paid[opp] - paid[me]
                stack = self.starting_stack - total[me]
                if tok == "c":
                    add = min(owe, stack)
                elif tok == "f":
                    add = 0.0
                elif tok == "a":
                    add = stack
                else:
                    add = self.raise_add(tok, si, owe, total[0] + total[1], paid[me])
                if tok not in ("c", "f") and add - owe >= inc - 1e-9:
                    inc = add - owe            # full raise sets the new increment
                paid[me] += add
                total[me] += add
                to_act = opp
        if len(memo) >= _MEMO_LIMIT:
            memo.clear()
        memo[streets] = inc
        return inc

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

    def _replay(self, state: HoldemState):
        """Memoized :meth:`_replay_uncached` (results are immutable tuples)."""
        memo = self._replay_memo
        hit = memo.get(state.streets)
        if hit is None:
            if len(memo) >= _MEMO_LIMIT:
                memo.clear()
            hit = memo[state.streets] = self._replay_uncached(state.streets)
        return hit

    def _replay_uncached(self, streets: Tuple[str, ...]):
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
        for street_idx, street_actions in enumerate(streets):
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
                    add = self.raise_add(tok, street_idx, owe, pot_now, street_paid[me])
                    n_raises += 1
                street_paid[me] += add
                total[me] += add
                to_act = 1 - to_act
        return tuple(street_paid), tuple(total), to_act, n_raises

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
        return self.encoder.encode(self, state)

    def legal_actions(self, state: HoldemState) -> List[str]:
        memo = self._legal_memo
        hit = memo.get(state.streets)
        if hit is None:
            if len(memo) >= _MEMO_LIMIT:
                memo.clear()
            hit = memo[state.streets] = tuple(self._legal_uncached(state))
        return list(hit)

    def _legal_uncached(self, state: HoldemState) -> List[str]:
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
            street = len(state.streets) - 1
            min_inc = self._min_increment(state.streets) if self.enforce_min_raise else 0.0
            for name in self.raise_tokens(street):
                add = self.raise_add(name, street, owe, pot_now, street_paid[me])
                if add - owe < min_inc - 1e-9:
                    continue               # below the NLHE minimum bet / raise
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
            delta = self.raise_add(action, len(state.streets) - 1, owe, pot_now, street_paid[me])

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
