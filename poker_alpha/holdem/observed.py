"""Hero-visible table state: the normalized input to analysis.

Every input adapter (manual entry, simulator, hand-history replay, screen
observer) produces an :class:`ObservedTableState`. It contains only what the
hero can see — never opponents' hole cards (except when revealed at
showdown, recorded separately as ``shown_cards``).

Amounts are in chips (floats, since screens may show decimals). The big blind
is part of the state so callers can convert to BB.

:func:`validate` returns every problem it finds instead of stopping at the
first, so a UI can show them all and a tracker can decide what to reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..poker.cards import card_str
from .positions import position_names
from .state import Street

_EPS = 1e-6

_BOARD_FOR_STREET = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4,
                     Street.RIVER: 5, Street.SHOWDOWN: 5}


@dataclass(frozen=True)
class ObservedSeat:
    seat: int
    occupied: bool = True
    name: str = ""
    stack: Optional[float] = None          # chips behind (None = unreadable)
    current_bet: float = 0.0               # this street's commitment
    committed_total: Optional[float] = None  # this hand, if known
    folded: bool = False
    all_in: bool = False
    sitting_out: bool = False

    @property
    def in_hand(self) -> bool:
        return self.occupied and not self.folded and not self.sitting_out


@dataclass(frozen=True)
class ObservedAction:
    street: int
    seat: int
    kind: str                  # fold/check/call/bet/raise/all_in/post
    amount: float = 0.0        # raise-to (bet/raise) or chips added (call/post)


@dataclass(frozen=True)
class ObservedTableState:
    num_seats: int
    hero_seat: int
    dealer: int
    seats: Tuple[ObservedSeat, ...]
    street: Street
    small_blind: float
    big_blind: float
    ante: float = 0.0
    hero_cards: Optional[Tuple[int, int]] = None
    board: Tuple[int, ...] = ()
    pot_total: Optional[float] = None      # all chips in the middle incl. current bets
    actor: Optional[int] = None
    action_history: Tuple[ObservedAction, ...] = ()
    shown_cards: Tuple[Tuple[int, Tuple[int, int]], ...] = ()
    hand_id: str = ""
    timestamp: Optional[float] = None
    source: str = ""

    # -- derived quantities ----------------------------------------------------

    def seat(self, i: int) -> ObservedSeat:
        return self.seats[i]

    @property
    def hero(self) -> ObservedSeat:
        return self.seats[self.hero_seat]

    @property
    def max_bet(self) -> float:
        return max((s.current_bet for s in self.seats if s.occupied), default=0.0)

    @property
    def amount_to_call(self) -> float:
        hero = self.hero
        owe = max(0.0, self.max_bet - hero.current_bet)
        if hero.stack is not None:
            owe = min(owe, hero.stack)
        return owe

    @property
    def pot(self) -> float:
        """Pot including current-street bets (falls back to the bets seen)."""
        if self.pot_total is not None:
            return self.pot_total
        known = [s.committed_total for s in self.seats if s.occupied]
        if all(k is not None for k in known):
            return float(sum(known))
        return float(sum(s.current_bet for s in self.seats))

    @property
    def opponents_in_hand(self) -> Tuple[int, ...]:
        return tuple(s.seat for s in self.seats
                     if s.in_hand and s.seat != self.hero_seat)

    @property
    def effective_stack(self) -> Optional[float]:
        """Hero's stack vs the largest live opponent stack (chips behind)."""
        hero = self.hero.stack
        opp = [self.seats[i].stack for i in self.opponents_in_hand]
        if hero is None or not opp or any(o is None for o in opp):
            return None
        return min(hero, max(opp))

    @property
    def pot_odds(self) -> Optional[float]:
        """Call / (pot after calling); None when nothing to call."""
        call = self.amount_to_call
        if call <= _EPS:
            return None
        return call / (self.pot + call)

    @property
    def spr(self) -> Optional[float]:
        eff = self.effective_stack
        if eff is None or self.pot <= 0:
            return None
        return eff / self.pot

    def positions(self) -> Dict[int, str]:
        """Position names over the occupied seats (clockwise from the button)."""
        occupied = [s.seat for s in self.seats if s.occupied and not s.sitting_out]
        if self.dealer not in occupied or len(occupied) < 2:
            return {}
        start = occupied.index(self.dealer)
        ordered = occupied[start:] + occupied[:start]
        names = position_names(len(ordered), 0)
        return {seat: names[i] for i, seat in enumerate(ordered)}

    @property
    def hero_position(self) -> Optional[str]:
        return self.positions().get(self.hero_seat)

    def to_bb(self, chips: Optional[float]) -> Optional[float]:
        return None if chips is None else chips / self.big_blind

    def describe(self) -> str:
        hc = " ".join(card_str(c) for c in self.hero_cards) if self.hero_cards else "??"
        board = " ".join(card_str(c) for c in self.board) or "-"
        return (f"{self.street.name} hero={self.hero_position or self.hero_seat} "
                f"[{hc}] board [{board}] pot {self.to_bb(self.pot):.1f}bb "
                f"to call {self.to_bb(self.amount_to_call):.1f}bb")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str    # "error" (impossible state) or "warning" (suspicious)
    code: str
    message: str


def validate(state: ObservedTableState) -> List[ValidationIssue]:
    """All detectable inconsistencies in ``state`` (empty list = valid)."""
    issues: List[ValidationIssue] = []

    def err(code: str, msg: str) -> None:
        issues.append(ValidationIssue("error", code, msg))

    def warn(code: str, msg: str) -> None:
        issues.append(ValidationIssue("warning", code, msg))

    n = state.num_seats
    if not 2 <= n <= 9:
        err("seat_count", f"num_seats {n} outside 2-9")
    if len(state.seats) != n:
        err("seat_count", f"{len(state.seats)} seat records for {n} seats")
        return issues
    if any(s.seat != i for i, s in enumerate(state.seats)):
        err("seat_index", "seat records out of order")
    if not 0 <= state.hero_seat < n or not state.seats[state.hero_seat].occupied:
        err("hero_seat", "hero seat is not an occupied seat")
    if not 0 <= state.dealer < n or not state.seats[state.dealer].occupied:
        err("dealer", "dealer button is not on an occupied seat")
    if state.big_blind <= 0 or state.small_blind < 0 or state.ante < 0:
        err("blinds", "blinds/ante must be non-negative and big blind positive")

    cards: List[int] = list(state.board)
    if state.hero_cards is not None:
        if len(state.hero_cards) != 2:
            err("hero_cards", "hero must hold exactly two cards")
        cards.extend(state.hero_cards)
        if set(state.hero_cards) & set(state.board):
            err("hero_board_collision", "hero cards also appear on the board")
    for _, shown in state.shown_cards:
        cards.extend(shown)
    if any(not 0 <= c < 52 for c in cards):
        err("card_range", "card code out of range")
    if len(cards) != len(set(cards)):
        err("duplicate_cards", "the same card appears more than once")
    expected = _BOARD_FOR_STREET.get(state.street)
    if len(state.board) not in (0, 3, 4, 5):
        err("board_size", f"board has {len(state.board)} cards")
    elif expected is not None and len(state.board) != expected:
        err("board_street", f"{state.street.name} needs {expected} board "
                            f"cards, saw {len(state.board)}")

    for s in state.seats:
        if not s.occupied:
            if s.current_bet > _EPS or (s.stack or 0) > _EPS:
                warn("empty_seat_chips", f"empty seat {s.seat} shows chips")
            continue
        if s.stack is not None and s.stack < -_EPS:
            err("negative_stack", f"seat {s.seat} stack {s.stack} < 0")
        if s.current_bet < -_EPS:
            err("negative_bet", f"seat {s.seat} bet {s.current_bet} < 0")
        if s.all_in and s.stack is not None and s.stack > _EPS:
            err("all_in_with_chips", f"seat {s.seat} all-in with chips behind")
        if s.committed_total is not None and s.committed_total + _EPS < s.current_bet:
            err("commitment", f"seat {s.seat} total commitment below street bet")

    bets = sum(s.current_bet for s in state.seats if s.occupied)
    if state.pot_total is not None:
        if state.pot_total + _EPS < bets:
            err("pot_below_bets", f"pot {state.pot_total} smaller than the "
                                  f"{bets} in current bets")
        known = [s.committed_total for s in state.seats if s.occupied]
        if known and all(k is not None for k in known) and \
                abs(sum(known) - state.pot_total) > 1e-4:
            err("pot_mismatch", f"pot {state.pot_total} != contributions "
                                f"{sum(known)}")
        if state.street == Street.PREFLOP and state.pot_total + _EPS < \
                min(state.big_blind, bets or state.big_blind):
            warn("pot_small", "preflop pot smaller than the big blind")

    if state.actor is not None:
        if not 0 <= state.actor < n:
            err("actor", "actor seat out of range")
        else:
            a = state.seats[state.actor]
            if not a.in_hand:
                err("actor", f"actor seat {state.actor} is not in the hand")
            elif a.all_in:
                err("actor", f"actor seat {state.actor} is all-in")
    live = [s for s in state.seats if s.in_hand]
    if len(live) < 2 and state.street != Street.SHOWDOWN:
        warn("hand_over", "fewer than two players remain; no decision to make")
    if state.hero.folded:
        warn("hero_folded", "hero has folded")
    return issues


def is_valid(state: ObservedTableState) -> bool:
    return not any(i.severity == "error" for i in validate(state))
