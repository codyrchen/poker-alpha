"""Input adapters producing :class:`ObservedTableState`.

* :class:`ManualStateAdapter` — from a plain, JSON-friendly dict (card
  strings like ``"As"``), e.g. typed by a user or loaded from a file. The
  same dict layout (with ``"format": "pokeralpha.observed/v1"``) is what
  :func:`observed_to_dict` writes, so it round-trips.
* :class:`SimulationStateAdapter` — from the rules engine's full
  :class:`HoldemTableState`, hiding every card the hero cannot see.

Adapters never invent information: fields that are absent stay ``None``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..poker.cards import card_code, card_str
from .observed import ObservedAction, ObservedSeat, ObservedTableState
from .state import HoldemTableState, Street

OBSERVED_FORMAT = "pokeralpha.observed/v1"


def parse_cards(spec: Any) -> tuple:
    """``"As Kd"``, ``"AsKd"``, ``["As", "Kd"]`` or codes -> tuple of codes."""
    if spec is None or spec == "":
        return ()
    if isinstance(spec, str):
        compact = spec.replace(" ", "").replace(",", "")
        if len(compact) % 2:
            raise ValueError(f"cannot parse cards {spec!r}")
        return tuple(card_code(compact[i:i + 2]) for i in range(0, len(compact), 2))
    out = []
    for c in spec:
        out.append(card_code(c) if isinstance(c, str) else int(c))
    return tuple(out)


def _street(value: Any, board_len: int) -> Street:
    if value is None:
        return {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN,
                5: Street.RIVER}.get(board_len, Street.PREFLOP)
    if isinstance(value, str):
        return Street[value.upper()]
    return Street(int(value))


class ManualStateAdapter:
    """Build an :class:`ObservedTableState` from a dict."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ObservedTableState:
        fmt = data.get("format", OBSERVED_FORMAT)
        if fmt != OBSERVED_FORMAT:
            raise ValueError(f"unsupported observed-state format {fmt!r}")
        seats_in: Sequence[Dict[str, Any]] = data["seats"]
        seats = []
        for i, s in enumerate(seats_in):
            seats.append(ObservedSeat(
                seat=i,
                occupied=bool(s.get("occupied", True)),
                name=str(s.get("name", "")),
                stack=None if s.get("stack") is None else float(s["stack"]),
                current_bet=float(s.get("bet", 0.0)),
                committed_total=None if s.get("committed") is None
                else float(s["committed"]),
                folded=bool(s.get("folded", False)),
                all_in=bool(s.get("all_in", False)),
                sitting_out=bool(s.get("sitting_out", False)),
            ))
        board = parse_cards(data.get("board"))
        hero = parse_cards(data.get("hero_cards"))
        actions = tuple(
            ObservedAction(street=int(_street(a.get("street"), 0)),
                           seat=int(a["seat"]), kind=str(a["kind"]),
                           amount=float(a.get("amount", 0.0)))
            for a in data.get("actions", ()))
        shown = tuple((int(seat), tuple(parse_cards(c)))
                      for seat, c in data.get("shown_cards", {}).items())
        return ObservedTableState(
            num_seats=int(data.get("num_seats", len(seats))),
            hero_seat=int(data["hero_seat"]),
            dealer=int(data["dealer"]),
            seats=tuple(seats),
            street=_street(data.get("street"), len(board)),
            small_blind=float(data["small_blind"]),
            big_blind=float(data["big_blind"]),
            ante=float(data.get("ante", 0.0)),
            hero_cards=hero if hero else None,
            board=board,
            pot_total=None if data.get("pot") is None else float(data["pot"]),
            actor=None if data.get("actor") is None else int(data["actor"]),
            action_history=actions,
            shown_cards=shown,
            hand_id=str(data.get("hand_id", "")),
            timestamp=data.get("timestamp"),
            source=str(data.get("source", "manual")),
        )


def observed_to_dict(state: ObservedTableState) -> Dict[str, Any]:
    """Versioned, JSON-serializable form (inverse of ``from_dict``)."""
    return {
        "format": OBSERVED_FORMAT,
        "num_seats": state.num_seats,
        "hero_seat": state.hero_seat,
        "dealer": state.dealer,
        "small_blind": state.small_blind,
        "big_blind": state.big_blind,
        "ante": state.ante,
        "street": state.street.name,
        "hero_cards": [card_str(c) for c in state.hero_cards] if state.hero_cards else None,
        "board": [card_str(c) for c in state.board],
        "pot": state.pot_total,
        "actor": state.actor,
        "seats": [{
            "occupied": s.occupied, "name": s.name, "stack": s.stack,
            "bet": s.current_bet, "committed": s.committed_total,
            "folded": s.folded, "all_in": s.all_in,
            "sitting_out": s.sitting_out,
        } for s in state.seats],
        "actions": [{"street": a.street, "seat": a.seat, "kind": a.kind,
                     "amount": a.amount} for a in state.action_history],
        "shown_cards": {str(seat): [card_str(c) for c in cards]
                        for seat, cards in state.shown_cards},
        "hand_id": state.hand_id,
        "timestamp": state.timestamp,
        "source": state.source,
    }


class SimulationStateAdapter:
    """Project the engine's omniscient state onto one hero's view."""

    @staticmethod
    def observe(state: HoldemTableState, hero_seat: int,
                timestamp: Optional[float] = None) -> ObservedTableState:
        seats = tuple(ObservedSeat(
            seat=s.seat, occupied=True, stack=float(s.stack),
            current_bet=float(s.committed_street),
            committed_total=float(s.committed_total),
            folded=s.folded, all_in=s.all_in) for s in state.seats)
        # All-ins are reported as "all_in" with the resulting street total
        # (what a viewer sees); bets/raises as raise-to; calls as chips added.
        actions = tuple(ObservedAction(
            street=r.street, seat=r.seat,
            kind="all_in" if r.all_in else r.type.value,
            amount=float(r.street_total if r.all_in
                         or r.type.value in ("bet", "raise") else r.added))
            for r in state.action_history)
        hero = state.seats[hero_seat].hole_cards
        street = state.street if state.street <= Street.SHOWDOWN else Street.SHOWDOWN
        return ObservedTableState(
            num_seats=state.num_seats, hero_seat=hero_seat,
            dealer=state.dealer, seats=seats, street=Street(street),
            small_blind=float(state.small_blind),
            big_blind=float(state.big_blind), ante=float(state.ante),
            hero_cards=tuple(hero) if hero else None, board=state.board,
            pot_total=float(state.pot), actor=state.actor,
            action_history=actions, hand_id=state.hand_id,
            timestamp=timestamp, source="simulation")
