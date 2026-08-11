"""Leduc Poker: a two-round poker game with public cards and raises.

Leduc hold'em (Southey et al. 2005) is the standard step up from Kuhn: still
small enough to solve exactly, but with the structural features of real poker —
a public (community) card, two betting rounds, raises, and pot growth.

Rules
-----
* Six-card deck: two suits x three ranks (J=0, Q=1, K=2). Suits never matter
  for hand strength; they only make pairing possible.
* Each player antes 1 chip and receives one private card.
* **Round 1**: betting with a fixed bet/raise size of 2 chips.
* A public card is then dealt from the four remaining cards.
* **Round 2**: betting with a fixed bet/raise size of 4 chips.
* At most two bet/raises per round. Player 0 acts first in both rounds.
* Showdown: a player whose private card pairs the public card wins; otherwise
  the higher private rank wins; equal ranks split (utility 0).

Actions (single characters in the history string, rounds separated by ``/``):

* ``f`` — fold (only legal facing a bet/raise)
* ``c`` — check (no outstanding bet) or call (facing a bet/raise)
* ``r`` — bet/raise (until the round's raise cap is reached)

The history grammar is small enough to parse directly; the parse of a betting
string is memoized since CFR revisits the same histories constantly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from .base import Game

_RANKS = 3
_DECK = tuple(range(6))  # card // 2 == rank, suits irrelevant to strength
_RAISE_CAP = 2
_BET_SIZE = {0: 2, 1: 4}  # per round
_ANTE = 1


def rank(card: int) -> int:
    return card // 2


@dataclass(frozen=True)
class LeducState:
    """Cards dealt so far plus the betting history.

    ``cards`` is ``()`` before the deal, else ``(card0, card1)``.
    ``public`` is ``None`` until the community card is dealt.
    ``history`` holds round-1 actions, then ``/``, then round-2 actions.
    """

    cards: Tuple[int, ...]
    public: Optional[int]
    history: str


@lru_cache(maxsize=None)
def _parse_round(actions: str, bet: int) -> Tuple[int, int, int, bool, bool]:
    """Parse one round's action string.

    Returns ``(paid0, paid1, raises, folded_player_or_-1, round_over)`` where
    ``paid`` are chips contributed this round and ``folded`` is encoded in the
    third slot of the tuple below.
    """
    paid = [0, 0]
    raises = 0
    to_act = 0
    outstanding = 0  # amount the player to act must add to continue
    folded = -1
    over = False
    for i, a in enumerate(actions):
        if a == "r":
            paid[to_act] += outstanding + bet
            outstanding = bet
            raises += 1
        elif a == "c":
            paid[to_act] += outstanding
            was_facing_bet = outstanding > 0
            outstanding = 0
            if was_facing_bet or i == 1:
                # A call closes the round; the second check closes it too.
                over = True
        elif a == "f":
            folded = to_act
            over = True
        else:
            raise ValueError(f"bad action {a!r}")
        to_act = 1 - to_act
    return paid[0], paid[1], raises, folded, over


@lru_cache(maxsize=None)
def _parse_history(history: str) -> Tuple[int, int, int, int, int, bool, int]:
    """Parse a full history.

    Returns ``(round, to_act, paid0, paid1, raises_this_round,
    betting_over, folded_player)`` where ``paid`` includes antes,
    ``betting_over`` means the *current* round's betting has concluded, and
    ``folded_player`` is -1 if nobody folded.
    """
    paid0, paid1 = _ANTE, _ANTE
    if "/" in history:
        r1, r2 = history.split("/", 1)
        current_round = 1
    else:
        r1, r2 = history, None
        current_round = 0

    p0, p1, raises, folded, over = _parse_round(r1, _BET_SIZE[0])
    paid0 += p0
    paid1 += p1
    to_act = len(r1) % 2
    if r2 is not None:
        p0, p1, raises, folded2, over = _parse_round(r2, _BET_SIZE[1])
        paid0 += p0
        paid1 += p1
        to_act = len(r2) % 2
        if folded2 != -1:
            folded = folded2
    return (current_round, to_act, paid0, paid1, raises, over, folded)


class LeducPoker(Game):
    """Leduc Poker as an extensive-form game (see module docstring)."""

    def root(self) -> LeducState:
        return LeducState(cards=(), public=None, history="")

    # -- chance ----------------------------------------------------------

    def is_chance(self, state: LeducState) -> bool:
        if len(state.cards) == 0:
            return True
        if state.public is None:
            # Public card is dealt once round-1 betting concludes without a fold.
            rnd, _, _, _, _, over, folded = _parse_history(state.history)
            return rnd == 0 and over and folded == -1
        return False

    def chance_outcomes(self, state: LeducState) -> List[Tuple[float, LeducState]]:
        if len(state.cards) == 0:
            deals = [(a, b) for a in _DECK for b in _DECK if a != b]
            p = 1.0 / len(deals)
            return [(p, LeducState((a, b), None, "")) for a, b in deals]
        remaining = [c for c in _DECK if c not in state.cards]
        p = 1.0 / len(remaining)
        return [
            (p, LeducState(state.cards, c, state.history + "/"))
            for c in remaining
        ]

    # -- terminals -------------------------------------------------------

    def is_terminal(self, state: LeducState) -> bool:
        if len(state.cards) == 0:
            return False
        rnd, _, _, _, _, over, folded = _parse_history(state.history)
        if folded != -1:
            return True
        # Showdown: round-2 betting concluded.
        return rnd == 1 and over

    def utility(self, state: LeducState) -> float:
        _, _, paid0, paid1, _, _, folded = _parse_history(state.history)
        if folded == 0:
            return -float(paid0)
        if folded == 1:
            return float(paid1)
        # Showdown.
        c0, c1 = state.cards
        pub = state.public
        pair0 = rank(c0) == rank(pub)
        pair1 = rank(c1) == rank(pub)
        if pair0 and not pair1:
            return float(paid1)
        if pair1 and not pair0:
            return -float(paid0)
        if rank(c0) > rank(c1):
            return float(paid1)
        if rank(c1) > rank(c0):
            return -float(paid0)
        return 0.0

    # -- decisions -------------------------------------------------------

    def current_player(self, state: LeducState) -> int:
        _, to_act, _, _, _, _, _ = _parse_history(state.history)
        return to_act

    def infoset_key(self, state: LeducState) -> str:
        player = self.current_player(state)
        pub = "?" if state.public is None else str(state.public // 2)
        # Rank alone identifies the private card strategically; suits are
        # interchangeable. Using rank merges strategically identical infosets.
        return f"{rank(state.cards[player])}|{pub}|{state.history}"

    def legal_actions(self, state: LeducState) -> List[str]:
        rnd, _, _, _, raises, _, _ = _parse_history(state.history)
        actions_this_round = state.history.split("/")[-1]
        facing_bet = self._facing_bet(actions_this_round)
        acts: List[str] = []
        if facing_bet:
            acts.append("f")
        acts.append("c")
        if raises < _RAISE_CAP:
            acts.append("r")
        return acts

    @staticmethod
    def _facing_bet(actions_this_round: str) -> bool:
        return actions_this_round.endswith("r")

    def next_state(self, state: LeducState, action: str) -> LeducState:
        return LeducState(state.cards, state.public, state.history + action)
