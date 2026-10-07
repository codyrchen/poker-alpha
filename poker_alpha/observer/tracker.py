"""Temporal smoothing of noisy per-frame readings and event inference.

:class:`FieldTracker` keeps one field's *stable* value. A new reading
replaces it only if it is high-confidence, or the same candidate value has
been read in ``confirm_frames`` consecutive usable frames — so one bad OCR
frame never overwrites a stable value. Readings below ``min_confidence`` are
ignored outright. Fields like hero cards use ``always_confirm`` (agreement
across frames is required even at high confidence).

:func:`infer_events` compares two stable snapshots and names what changed
(bet placed, stack dropped, bets swept into the pot, fold, board card dealt,
dealer moved / new hand).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pokernow import FieldReading, FrameObservation


@dataclass
class FieldTracker:
    name: str
    confirm_frames: int = 2
    high_confidence: float = 0.9
    min_confidence: float = 0.3
    always_confirm: bool = False
    stable: Any = None
    stable_confidence: float = 0.0
    stable_since: Optional[float] = None
    has_value: bool = False
    candidate: Any = None
    candidate_count: int = 0
    rejected: int = 0
    pinned: bool = False            # manual override in force

    def update(self, reading: FieldReading) -> bool:
        """Feed one reading; return True if the stable value changed."""
        if self.pinned:
            return False
        if reading.confidence < self.min_confidence:
            self.rejected += 1
            return False
        if self.has_value and reading.value == self.stable:
            self.stable_confidence = max(self.stable_confidence * 0.5,
                                         reading.confidence)
            self.candidate, self.candidate_count = None, 0
            return False
        if reading.value == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = reading.value, 1
        fast = reading.confidence >= self.high_confidence and not self.always_confirm
        if fast or self.candidate_count >= self.confirm_frames:
            return self.accept(reading.value, reading.confidence, reading.timestamp)
        return False

    def accept(self, value, confidence: float, timestamp=None) -> bool:
        changed = not self.has_value or value != self.stable
        self.stable, self.stable_confidence = value, confidence
        self.stable_since = timestamp
        self.has_value = True
        self.candidate, self.candidate_count = None, 0
        return changed

    def pin(self, value) -> None:
        self.accept(value, 1.0)
        self.pinned = True

    def unpin(self) -> None:
        self.pinned = False


@dataclass(frozen=True)
class TableSnapshot:
    """Stable (smoothed) table facts at one moment."""

    pot: Optional[float]
    board: Tuple[str, ...]
    hero_cards: Tuple[Optional[str], Optional[str]]
    dealer: Optional[int]
    actor: Optional[int]
    stacks: Tuple[Optional[float], ...]
    bets: Tuple[float, ...]
    in_hand: Tuple[bool, ...]
    occupied: Tuple[bool, ...]
    all_in: Tuple[bool, ...]
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class TableEvent:
    kind: str           # bet / stack_decrease / bets_collected / fold / board / new_hand / stack_increase
    seat: Optional[int] = None
    amount: Optional[float] = None
    detail: str = ""


def infer_events(prev: TableSnapshot, cur: TableSnapshot) -> List[TableEvent]:
    events: List[TableEvent] = []
    new_hand = (prev.dealer is not None and cur.dealer is not None
                and prev.dealer != cur.dealer)
    if new_hand or (len(cur.board) < len(prev.board)):
        events.append(TableEvent("new_hand", detail="dealer moved"
                                  if new_hand else "board cleared"))
        return events
    if len(cur.board) > len(prev.board):
        events.append(TableEvent("board", detail=" ".join(cur.board[len(prev.board):])))
    if sum(cur.bets) < sum(prev.bets) - 1e-9 and (cur.pot or 0) > (prev.pot or 0):
        events.append(TableEvent("bets_collected",
                                 amount=(cur.pot or 0) - (prev.pot or 0)))
    for s in range(len(cur.stacks)):
        if prev.in_hand[s] and not cur.in_hand[s]:
            events.append(TableEvent("fold", seat=s))
        if cur.bets[s] > prev.bets[s] + 1e-9:
            events.append(TableEvent("bet", seat=s, amount=cur.bets[s]))
        a, b = prev.stacks[s], cur.stacks[s]
        if a is not None and b is not None:
            if b < a - 1e-9:
                events.append(TableEvent("stack_decrease", seat=s, amount=a - b))
            elif b > a + 1e-9:
                events.append(TableEvent("stack_increase", seat=s, amount=b - a))
    return events
