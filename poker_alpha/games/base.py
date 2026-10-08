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

import numpy as np

State = Hashable
Action = str


class Game(ABC):
    """A two-player zero-sum extensive-form game with chance nodes."""

    num_players: int = 2

    def signature(self) -> str:
        """Deterministic, versioned identifier of this game's rules/config.

        Stored in solver checkpoints so a checkpoint cannot be resumed against
        a different game. Games with configuration must override this and
        encode every parameter that changes the game tree.
        """
        return f"{type(self).__name__}:v1"

    def solver_config_signature(self) -> str:
        """Signature of the locked solver configuration this game was built
        from (``HoldemSolverConfig.build_game``), or ``""``."""
        cfg = getattr(self, "solver_config", None)
        return cfg.signature() if cfg is not None else ""

    def encoder_signature(self) -> "str | None":
        """Signature of the information-state encoder, if the game has one.

        ``None`` means information-set keys are the game's built-in keys.
        """
        return None

    @abstractmethod
    def root(self) -> State:
        """Return the initial state (typically a chance node)."""

    @abstractmethod
    def is_chance(self, state: State) -> bool:
        """True if ``state`` is a chance node (e.g. a deal)."""

    @abstractmethod
    def chance_outcomes(self, state: State) -> List[Tuple[float, State]]:
        """Return ``(probability, successor)`` pairs for a chance node.

        Games whose chance trees are too large to enumerate (Hold'em) may
        raise :class:`NotImplementedError` here and override
        :meth:`sample_chance` instead; exact full-tree algorithms (CFR, CFR+,
        exact evaluation) then do not apply to them.
        """

    def sample_chance(self, state: State, rng: np.random.Generator) -> State:
        """Sample one chance successor of ``state`` using ``rng``.

        The default samples from :meth:`chance_outcomes` with exactly one
        ``rng.choice(n, p=...)`` call, so sampling solvers consume the RNG
        stream identically to the pre-existing enumerating implementation
        (seeded Kuhn/Leduc MCCFR results are unchanged). Games that cannot
        enumerate their chance outcomes override this.
        """
        outcomes = self.chance_outcomes(state)
        probs = np.array([p for p, _ in outcomes])
        idx = int(rng.choice(len(probs), p=probs / probs.sum()))
        return outcomes[idx][1]

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
