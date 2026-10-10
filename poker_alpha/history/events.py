"""Canonical, source-independent hand-history events.

Every hand-history source (the PokerAlpha JSON format, the simulator, a
screen-observer session, future site-specific parsers) is converted into this
event list; nothing downstream knows where a hand came from.

Amounts are in chips (floats allowed). ``to`` is the *street raise-to*
total for bets/raises/all-ins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type


@dataclass(frozen=True)
class Event:
    @property
    def type(self) -> str:
        return type(self).__name__

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = list(v) if isinstance(v, tuple) else v
        return d


@dataclass(frozen=True)
class SeatInfo:
    seat: int
    name: str
    stack: float


@dataclass(frozen=True)
class HandStarted(Event):
    hand_id: str
    num_seats: int
    dealer: int
    small_blind: float
    big_blind: float
    seats: Tuple[SeatInfo, ...]
    hero_seat: Optional[int] = None
    ante: float = 0.0
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["seats"] = [asdict(s) for s in self.seats]
        return d


@dataclass(frozen=True)
class BlindPosted(Event):
    seat: int
    amount: float
    blind: str  # "sb" or "bb"


@dataclass(frozen=True)
class AntePosted(Event):
    seat: int
    amount: float


@dataclass(frozen=True)
class CardDealt(Event):
    """Hole cards becoming known (to the hero, or shown)."""

    seat: int
    cards: Tuple[str, str]


@dataclass(frozen=True)
class FlopDealt(Event):
    cards: Tuple[str, str, str]


@dataclass(frozen=True)
class TurnDealt(Event):
    card: str


@dataclass(frozen=True)
class RiverDealt(Event):
    card: str


@dataclass(frozen=True)
class PlayerChecked(Event):
    seat: int


@dataclass(frozen=True)
class PlayerCalled(Event):
    seat: int
    amount: Optional[float] = None   # chips added (informational)


@dataclass(frozen=True)
class PlayerBet(Event):
    seat: int
    to: float


@dataclass(frozen=True)
class PlayerRaised(Event):
    seat: int
    to: float


@dataclass(frozen=True)
class PlayerFolded(Event):
    seat: int


@dataclass(frozen=True)
class PlayerAllIn(Event):
    seat: int
    to: Optional[float] = None       # street total after the all-in (informational)


@dataclass(frozen=True)
class Showdown(Event):
    seat: int
    cards: Tuple[str, str]


@dataclass(frozen=True)
class PotAwarded(Event):
    seat: int
    amount: float


EVENT_TYPES: Dict[str, Type[Event]] = {
    cls.__name__: cls for cls in (
        HandStarted, BlindPosted, AntePosted, CardDealt, FlopDealt, TurnDealt,
        RiverDealt, PlayerChecked, PlayerCalled, PlayerBet, PlayerRaised,
        PlayerFolded, PlayerAllIn, Showdown, PotAwarded)
}

ACTION_EVENTS = (PlayerChecked, PlayerCalled, PlayerBet, PlayerRaised,
                 PlayerFolded, PlayerAllIn)


def event_from_dict(d: Dict[str, Any]) -> Event:
    t = d.get("type")
    cls = EVENT_TYPES.get(t)
    if cls is None:
        raise ValueError(f"unknown event type {t!r}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    names = {f.name for f in fields(cls)}
    unknown = set(kwargs) - names
    if unknown:
        raise ValueError(f"{t}: unknown fields {sorted(unknown)}")
    if cls is HandStarted:
        kwargs["seats"] = tuple(SeatInfo(int(s["seat"]), str(s.get("name", "")),
                                         float(s["stack"])) for s in kwargs["seats"])
    for k in ("cards",):
        if k in kwargs:
            kwargs[k] = tuple(kwargs[k])
    return cls(**kwargs)
