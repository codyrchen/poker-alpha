"""Sealed-bid auction duel: a non-poker hidden-information game.

A deliberately small game showing that the solver stack is not
poker-specific. Each player receives a PRIVATE value for an item
(uniform over ``0..n_values-1``, independent), then both submit a sealed
bid from the same range — modelled sequentially, with player 1's
information set hiding player 0's bid, which makes the bids effectively
simultaneous.

The framework's solvers require two-player ZERO-SUM games, so this is the
**duel** form of a first-price auction: player 0's utility is the
*difference of surpluses*,

    u0 = surplus0 - surplus1,   surplus_i = (value_i - bid_i) if i wins else 0

with ties awarding the item to nobody. This is NOT the classic
general-sum first-price auction (whose equilibrium analysis differs);
it is a competitive-bidding game with hidden private values, chosen
because it is exactly solvable, CFR-compatible, and still exhibits the
interesting behavior — bids shade below value, and higher types bid
more.

Infoset keys: ``A<value>`` for player 0, ``B<value>`` for player 1
(player 1 never sees player 0's bid before acting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .base import Game


@dataclass(frozen=True)
class AuctionState:
    values: Tuple[int, ...]      # () before the deal, else (v0, v1)
    bids: Tuple[int, ...]        # (), (b0,), or (b0, b1)


class AuctionDuel(Game):
    """Zero-sum sealed-bid auction duel (see module docstring)."""

    def __init__(self, n_values: int = 3, n_bids: int = 3) -> None:
        self.n_values = n_values
        self.n_bids = n_bids

    def signature(self) -> str:
        return f"AuctionDuel:v1:values={self.n_values}:bids={self.n_bids}"

    def root(self) -> AuctionState:
        return AuctionState((), ())

    def is_chance(self, state: AuctionState) -> bool:
        return not state.values

    def chance_outcomes(self, state: AuctionState):
        p = 1.0 / (self.n_values ** 2)
        return [(p, AuctionState((v0, v1), ()))
                for v0 in range(self.n_values)
                for v1 in range(self.n_values)]

    def is_terminal(self, state: AuctionState) -> bool:
        return len(state.bids) == 2

    def utility(self, state: AuctionState) -> float:
        v0, v1 = state.values
        b0, b1 = state.bids
        s0 = (v0 - b0) if b0 > b1 else 0.0
        s1 = (v1 - b1) if b1 > b0 else 0.0
        return float(s0 - s1)

    def current_player(self, state: AuctionState) -> int:
        return len(state.bids)

    def infoset_key(self, state: AuctionState) -> str:
        player = self.current_player(state)
        # Neither player sees the other's bid before acting: player 1's key
        # deliberately excludes b0.
        return f"{'AB'[player]}{state.values[player]}"

    def legal_actions(self, state: AuctionState) -> List[str]:
        return [str(b) for b in range(self.n_bids)]

    def next_state(self, state: AuctionState, action: str) -> AuctionState:
        return AuctionState(state.values, state.bids + (int(action),))

    # -- complexity inspection ------------------------------------------------

    def complexity(self) -> dict:
        nv, nb = self.n_values, self.n_bids
        return {
            "players": 2,
            "chance_outcomes": nv * nv,
            "infosets": 2 * nv,
            "terminal_states": nv * nv * nb * nb,
            "branching_factor": nb,
            "private_information": "each player's item value",
        }


__all__ = ["AuctionDuel", "AuctionState"]
