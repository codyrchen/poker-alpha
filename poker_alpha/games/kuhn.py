"""Kuhn Poker: the smallest interesting imperfect-information poker game.

Rules
-----
* Deck of three cards J < Q < K, encoded 0 < 1 < 2.
* Two players; each antes 1 chip and is dealt one private card.
* One betting round with two actions:

  - ``p`` = pass (check, or fold when facing a bet),
  - ``b`` = bet (or call when facing a bet).

Terminal histories and player-0 payoffs (in chips):

===========  ===========================  ===========================
history      meaning                      utility to player 0
===========  ===========================  ===========================
``pp``       check-check, showdown        +1 / -1 by card
``bp``       bet, fold                    +1  (player 1 folds)
``bb``       bet, call, showdown          +2 / -2 by card
``pbp``      check, bet, fold             -1  (player 0 folds)
``pbb``      check, bet, call, showdown   +2 / -2 by card
===========  ===========================  ===========================

The game has a known Nash value of -1/18 to player 0 and a one-parameter family
of equilibria, which makes it the standard first test of a CFR implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .base import Game

_PASS = "p"
_BET = "b"
_ACTIONS = (_PASS, _BET)
_TERMINAL = {"pp", "bp", "bb", "pbp", "pbb"}


@dataclass(frozen=True)
class KuhnState:
    """A Kuhn game state: the dealt cards (empty before the deal) and history."""

    cards: Tuple[int, ...]  # () at the root chance node, else (card0, card1)
    history: str  # sequence of 'p'/'b'


class KuhnPoker(Game):
    """Kuhn Poker as an extensive-form game (see module docstring)."""

    n_cards = 3

    def root(self) -> KuhnState:
        return KuhnState(cards=(), history="")

    def is_chance(self, state: KuhnState) -> bool:
        return len(state.cards) == 0

    def chance_outcomes(self, state: KuhnState) -> List[Tuple[float, KuhnState]]:
        # Deal two distinct cards; all 6 ordered pairs are equally likely.
        deals = [
            (i, j)
            for i in range(self.n_cards)
            for j in range(self.n_cards)
            if i != j
        ]
        prob = 1.0 / len(deals)
        return [(prob, KuhnState(cards=(i, j), history="")) for i, j in deals]

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in _TERMINAL

    def utility(self, state: KuhnState) -> float:
        h = state.history
        c0, c1 = state.cards
        player0_wins = c0 > c1
        if h == "bp":
            return 1.0  # player 1 folded
        if h == "pbp":
            return -1.0  # player 0 folded
        if h == "pp":
            return 1.0 if player0_wins else -1.0
        if h in ("bb", "pbb"):
            return 2.0 if player0_wins else -2.0
        raise ValueError(f"utility() called on non-terminal history {h!r}")

    def current_player(self, state: KuhnState) -> int:
        return len(state.history) % 2

    def infoset_key(self, state: KuhnState) -> str:
        player = self.current_player(state)
        return f"{state.cards[player]}{state.history}"

    def legal_actions(self, state: KuhnState) -> List[str]:
        return list(_ACTIONS)

    def next_state(self, state: KuhnState, action: str) -> KuhnState:
        return KuhnState(cards=state.cards, history=state.history + action)
