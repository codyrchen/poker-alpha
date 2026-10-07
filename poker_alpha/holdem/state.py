"""Immutable multiplayer Hold'em table state (integer chips)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple

from .action import ActionRecord


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4     # betting finished; awaiting/after showdown
    COMPLETE = 5     # pots awarded


BOARD_SIZE = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4,
              Street.RIVER: 5}


@dataclass(frozen=True)
class SeatState:
    seat: int
    stack: int                 # chips behind
    committed_street: int = 0
    committed_total: int = 0   # this hand, blinds and antes included
    folded: bool = False
    all_in: bool = False
    hole_cards: Optional[Tuple[int, int]] = None

    @property
    def live(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in


@dataclass(frozen=True)
class TableConfig:
    small_blind: int
    big_blind: int
    ante: int = 0

    def __post_init__(self) -> None:
        for name in ("small_blind", "big_blind", "ante"):
            v = getattr(self, name)
            if int(v) != v or v < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.big_blind <= 0:
            raise ValueError("big_blind must be positive")


@dataclass(frozen=True)
class HoldemTableState:
    """Complete state of one hand in progress.

    Invariants (checked by :func:`~poker_alpha.holdem.engine.check_invariants`):
    ``sum(stack) + pot == starting chips`` until settlement; afterwards
    ``pot == 0`` and the chips sit in the stacks.
    """

    num_seats: int
    dealer: int
    config: TableConfig
    street: Street
    board: Tuple[int, ...]
    seats: Tuple[SeatState, ...]
    actor: Optional[int]           # seat to act, None between rounds/after
    current_bet: int               # highest street commitment to match
    min_raise: int                 # last full raise increment (>= big blind)
    last_full_bet: int             # street commitment of the last full bet/raise
    acted_level: Tuple[Optional[int], ...]  # last_full_bet when each seat last acted
    to_act: Tuple[int, ...]        # seats that still owe an action this round
    action_history: Tuple[ActionRecord, ...] = ()
    awards: Tuple[Tuple[int, int], ...] = ()   # (seat, chips won) at COMPLETE
    hand_id: str = ""

    @property
    def pot(self) -> int:
        """Chips in the middle (all commitments not yet awarded)."""
        if self.street == Street.COMPLETE:
            return 0
        return sum(s.committed_total for s in self.seats)

    @property
    def small_blind(self) -> int:
        return self.config.small_blind

    @property
    def big_blind(self) -> int:
        return self.config.big_blind

    @property
    def ante(self) -> int:
        return self.config.ante

    def amount_to_call(self, seat: Optional[int] = None) -> int:
        seat = self.actor if seat is None else seat
        if seat is None:
            return 0
        s = self.seats[seat]
        return max(0, min(self.current_bet - s.committed_street, s.stack))

    @property
    def amount_to_call_for_actor(self) -> int:
        return self.amount_to_call()

    @property
    def live_seats(self) -> Tuple[int, ...]:
        return tuple(s.seat for s in self.seats if s.live)

    @property
    def is_complete(self) -> bool:
        return self.street == Street.COMPLETE
