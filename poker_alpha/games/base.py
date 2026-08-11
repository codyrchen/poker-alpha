"""Extensive-form game interface for CFR-style solving.

The solvers in this project operate on any two-player, zero-sum, imperfect-
information game that exposes the small interface below. Keeping the interface
minimal (and utilities always expressed from *player 0's* perspective) is what
lets one CFR implementation train on Kuhn, Leduc, and later abstractions
without change.

Conventions
-----------
* Two players, indexed 0 and 1.
* Zero-sum: player 1's utility is always the negation of player 0's, so only
  :meth:`Game.utility` (player 0) is defined.
* Chance (card dealing, public cards) is modelled as explicit chance nodes with
  a probability distribution over successor states, including the root.
* An *information set* is identified by a string key that encodes exactly what
  the acting player knows (their private card(s) plus the public betting
  history) and nothing they do not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, List, Tuple

State = Hashable
Action = str


class Game(ABC):
    """A two-player zero-sum extensive-form game with chance nodes."""

    num_players: int = 2

    @abstractmethod
    def root(self) -> State:
        """Return the initial state (typically a chance node)."""

    @abstractmethod
    def is_chance(self, state: State) -> bool:
        """True if ``state`` is a chance node (e.g. a deal)."""

    @abstractmethod
    def chance_outcomes(self, state: State) -> List[Tuple[float, State]]:
        """Return ``(probability, successor)`` pairs for a chance node."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """True if ``state`` is terminal (the hand is over)."""

    @abstractmethod
    def utility(self, state: State) -> float:
        """Terminal utility for **player 0** (player 1 gets the negation)."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Index (0 or 1) of the player to act at a decision node."""

    @abstractmethod
    def infoset_key(self, state: State) -> str:
        """Information-set key for the acting player at ``state``."""

    @abstractmethod
    def legal_actions(self, state: State) -> List[Action]:
        """Legal actions at a decision node, in a deterministic order."""

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        """Return the successor state after ``action``."""
