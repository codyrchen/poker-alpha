"""Concrete Hold'em actions and the legal-action descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class Action:
    """A requested action. ``amount`` is the *raise-to* street commitment for
    BET/RAISE and ignored otherwise (ALL_IN commits the whole stack)."""

    type: ActionType
    amount: Optional[int] = None

    @staticmethod
    def fold() -> "Action":
        return Action(ActionType.FOLD)

    @staticmethod
    def check() -> "Action":
        return Action(ActionType.CHECK)

    @staticmethod
    def call() -> "Action":
        return Action(ActionType.CALL)

    @staticmethod
    def bet(to: int) -> "Action":
        return Action(ActionType.BET, int(to))

    @staticmethod
    def raise_to(to: int) -> "Action":
        return Action(ActionType.RAISE, int(to))

    @staticmethod
    def all_in() -> "Action":
        return Action(ActionType.ALL_IN)


@dataclass(frozen=True)
class ActionRecord:
    """What actually happened, as stored in the hand history."""

    street: int
    seat: int
    type: ActionType          # resolved type (an ALL_IN request becomes CALL/BET/RAISE + all_in)
    added: int                # chips moved from stack this action
    street_total: int         # seat's street commitment afterwards
    all_in: bool
    full_raise: bool = False  # True if this bet/raise reopened the action


@dataclass(frozen=True)
class LegalActions:
    """What the actor may do. Raise bounds are raise-to street commitments."""

    seat: int
    can_fold: bool
    can_check: bool
    call_amount: int           # chips to add to call (0 if checking)
    call_is_all_in: bool
    can_raise: bool            # bet (if no bet yet) or raise
    min_raise_to: int
    max_raise_to: int          # all-in level
    is_bet: bool               # True if no outstanding bet (BET vs RAISE)
