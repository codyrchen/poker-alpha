"""Replay canonical events through the rules engine.

``replay_hand(events)`` reconstructs every table state, validating the
history against the rules (out-of-turn actions, illegal sizes, wrong blinds,
duplicate cards and award mismatches raise :class:`ReplayError`). It
returns hero-view snapshots, every hero decision point with the action the
hero actually took, the final awards, and a
:class:`~poker_alpha.opponent.statistics.HandSummary` for player statistics.

Chips are converted to the engine's integer units with ``chip_scale``
(default 100, i.e. cents); amounts that are not exact multiples of
``1/chip_scale`` are rejected rather than silently rounded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..holdem import (Action, IllegalActionError, ObservedTableState,
                      SimulationStateAdapter, TableConfig, apply_action,
                      cards_needed, deal_board, position_names, settle,
                      start_hand)
from ..holdem.positions import blind_seats
from ..holdem.state import HoldemTableState, Street
from ..opponent.statistics import HandSummary
from ..poker.cards import card_code
from .events import (ACTION_EVENTS, AntePosted, BlindPosted, CardDealt, Event,
                     FlopDealt, HandStarted, PlayerAllIn, PlayerBet,
                     PlayerCalled, PlayerChecked, PlayerFolded, PlayerRaised,
                     PotAwarded, RiverDealt, Showdown, TurnDealt)


class ReplayError(ValueError):
    def __init__(self, index: int, message: str) -> None:
        super().__init__(f"event {index}: {message}")
        self.index = index


@dataclass(frozen=True)
class HeroDecision:
    event_index: int
    state: ObservedTableState
    action: str                 # check/call/bet/raise/fold/all_in
    amount: Optional[float]     # raise-to for bet/raise/all-in


@dataclass(frozen=True)
class ReplayResult:
    hand_id: str
    started: HandStarted
    snapshots: Tuple[Tuple[int, ObservedTableState], ...]
    decisions: Tuple[HeroDecision, ...]
    final_state: HoldemTableState
    awards: Dict[int, float]
    net: Dict[int, float]       # chips won minus chips put in, per seat
    summary: HandSummary
    warnings: Tuple[str, ...]


def replay_hand(events: Sequence[Event], hero_seat: Optional[int] = None,
                chip_scale: int = 100, hand_index: int = 0) -> ReplayResult:
    if not events or not isinstance(events[0], HandStarted):
        raise ReplayError(0, "hand must begin with HandStarted")
    start: HandStarted = events[0]
    hero = start.hero_seat if hero_seat is None else hero_seat
    warnings: List[str] = []
    unit = 1.0 / chip_scale

    def chips(x: float, i: int) -> int:
        v = x * chip_scale
        if abs(v - round(v)) > 1e-6:
            raise ReplayError(i, f"amount {x} not a multiple of {unit:g}")
        return int(round(v))

    n = start.num_seats
    by_seat = {s.seat: s for s in start.seats}
    if sorted(by_seat) != list(range(n)):
        raise ReplayError(0, "seats must be numbered 0..num_seats-1, all occupied")
    names = [by_seat[i].name for i in range(n)]
    known_holes: Dict[int, Tuple[int, int]] = {}
    for i, e in enumerate(events):
        if isinstance(e, CardDealt):
            if hero is not None and e.seat != hero:
                continue  # revealed later via Showdown; never leak early
            known_holes[e.seat] = tuple(card_code(c) for c in e.cards)
        elif isinstance(e, ACTION_EVENTS + (FlopDealt,)):
            break
    config = TableConfig(chips(start.small_blind, 0), chips(start.big_blind, 0),
                         chips(start.ante, 0))
    try:
        state = start_hand([chips(by_seat[i].stack, 0) for i in range(n)],
                           start.dealer, config,
                           hole_cards=[known_holes.get(i) for i in range(n)],
                           hand_id=start.hand_id)
    except ValueError as exc:
        raise ReplayError(0, str(exc)) from exc
    starting = {s.seat: s.stack + s.committed_total for s in state.seats}
    sb, bb = blind_seats(n, start.dealer)

    def observe(st: HoldemTableState) -> Optional[ObservedTableState]:
        # Settled hands are not table states anyone decides in; the result
        # is reported through ``awards``/``net`` instead.
        if hero is None or st.is_complete:
            return None
        return SimulationStateAdapter.observe(
            st, hero, timestamp=start.timestamp, chip_unit=unit, names=names,
            source="replay")

    snapshots: List[Tuple[int, ObservedTableState]] = []
    decisions: List[HeroDecision] = []
    awards_ev: Dict[int, float] = {}
    shown: Dict[int, Tuple[int, int]] = {}
    for i, e in enumerate(events[1:], start=1):
        if isinstance(e, BlindPosted):
            seat = sb if e.blind == "sb" else bb
            posted = state.seats[e.seat].committed_total - (config.ante and min(
                config.ante, starting[e.seat]))
            if e.seat != seat or chips(e.amount, i) != posted:
                raise ReplayError(i, f"{e.blind} post by seat {e.seat} of "
                                     f"{e.amount} disagrees with the rules")
        elif isinstance(e, AntePosted):
            if chips(e.amount, i) != min(config.ante, starting[e.seat]):
                raise ReplayError(i, "ante amount disagrees with HandStarted")
        elif isinstance(e, CardDealt):
            continue
        elif isinstance(e, (FlopDealt, TurnDealt, RiverDealt)):
            cards = e.cards if isinstance(e, FlopDealt) else (e.card,)
            try:
                state = deal_board(state, [card_code(c) for c in cards])
            except (IllegalActionError, ValueError) as exc:
                raise ReplayError(i, str(exc)) from exc
            obs = observe(state)
            if obs is not None:
                snapshots.append((i, obs))
        elif isinstance(e, ACTION_EVENTS):
            if state.actor != e.seat:
                raise ReplayError(i, f"seat {e.seat} acted out of turn "
                                     f"(expected {state.actor})")
            action, kind, amount = _to_action(e, i, chips)
            if e.seat == hero:
                obs = observe(state)
                decisions.append(HeroDecision(i, obs, kind, amount))
            try:
                state = apply_action(state, action)
            except IllegalActionError as exc:
                raise ReplayError(i, str(exc)) from exc
            obs = observe(state)
            if obs is not None:
                snapshots.append((i, obs))
        elif isinstance(e, Showdown):
            cards = tuple(card_code(c) for c in e.cards)
            shown[e.seat] = cards
            seats = list(state.seats)
            if seats[e.seat].hole_cards not in (None, cards):
                raise ReplayError(i, f"seat {e.seat} shows different cards")
            seats[e.seat] = replace(seats[e.seat], hole_cards=cards)
            state = replace(state, seats=tuple(seats))
        elif isinstance(e, PotAwarded):
            awards_ev[e.seat] = awards_ev.get(e.seat, 0.0) + e.amount
        else:
            raise ReplayError(i, f"unexpected event {e.type}")

    if not state.is_complete:
        if state.street == Street.SHOWDOWN or len(state.live_seats) == 1:
            try:
                state = settle(state)
            except IllegalActionError as exc:
                if not awards_ev:
                    raise ReplayError(len(events), f"cannot settle: {exc}") from exc
                warnings.append("showdown cards incomplete; using PotAwarded events")
        elif cards_needed(state) or state.actor is not None:
            raise ReplayError(len(events), "hand history ends mid-hand")
    pot_total = sum(s.committed_total for s in state.seats) * unit
    if state.is_complete:
        awards = {s: w * unit for s, w in state.awards}
        if awards_ev:
            for seat in set(awards) | set(awards_ev):
                if abs(awards.get(seat, 0.0) - awards_ev.get(seat, 0.0)) > 1e-6:
                    raise ReplayError(len(events), f"PotAwarded for seat {seat} "
                                      f"({awards_ev.get(seat, 0.0)}) disagrees "
                                      f"with the rules ({awards.get(seat, 0.0)})")
    else:
        awards = dict(awards_ev)
        if abs(sum(awards.values()) - pot_total) > 1e-6:
            raise ReplayError(len(events), "awards do not add up to the pot")
    net = {s.seat: awards.get(s.seat, 0.0) - s.committed_total * unit
           for s in state.seats}

    positions = position_names(n, start.dealer)
    acts = [_obs_action(r, unit) for r in state.action_history]
    showdown_seats = tuple(sorted(shown)) if shown else (
        tuple(state.live_seats) if len(state.live_seats) > 1 else ())
    summary = HandSummary(
        hand_index=hand_index, players={i: names[i] or f"seat{i}" for i in range(n)},
        positions=positions, actions=acts, big_blind_seat=bb,
        big_blind=start.big_blind, showdown_seats=showdown_seats,
        won={s: a for s, a in awards.items() if s in showdown_seats})
    return ReplayResult(start.hand_id, start, tuple(snapshots), tuple(decisions),
                        state, awards, net, summary, tuple(warnings))


def _to_action(e, i, chips):
    if isinstance(e, PlayerChecked):
        return Action.check(), "check", None
    if isinstance(e, PlayerCalled):
        return Action.call(), "call", None
    if isinstance(e, PlayerFolded):
        return Action.fold(), "fold", None
    if isinstance(e, PlayerBet):
        return Action.bet(chips(e.to, i)), "bet", e.to
    if isinstance(e, PlayerRaised):
        return Action.raise_to(chips(e.to, i)), "raise", e.to
    if isinstance(e, PlayerAllIn):
        return Action.all_in(), "all_in", e.to
    raise ReplayError(i, f"not an action: {e.type}")


@dataclass(frozen=True)
class _Act:
    street: int
    seat: int
    kind: str
    amount: float


def _obs_action(r, unit) -> _Act:
    kind = r.type.value
    amount = (r.street_total if kind in ("bet", "raise") else r.added) * unit
    return _Act(r.street, r.seat, kind, amount)
