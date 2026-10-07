"""No-limit Texas Hold'em rules engine for 2-9 players.

A pure, deterministic state machine over :class:`HoldemTableState` with
integer chips. It is independent of every solver: it knows the rules, not
strategy.

Flow::

    state = start_hand(stacks, dealer, config, hole_cards)
    while not state.is_complete:
        if state.actor is not None:
            state = apply_action(state, Action.call())   # or any legal action
        elif cards_needed(state):
            state = deal_board(state, next_cards)          # explicit or random
        else:
            state = settle(state)                          # showdown

Rules implemented
-----------------
* Antes (dead, not part of the street bet), small and big blind; blinds and
  antes larger than a stack are posted all-in.
* Heads-up: the button posts the small blind, acts first preflop and last
  postflop. 3+ handed: SB/BB follow the button, UTG acts first preflop and
  the first live seat left of the button first postflop.
* fold / check / call / bet / raise / all-in, with the minimum bet of one big
  blind and the minimum raise equal to the last *full* raise increment.
* Short all-ins: an all-in raise smaller than a full raise does not reopen the
  betting for players who already acted, unless several short raises
  *cumulatively* reach a full raise over the level that player last faced
  (TDA rule). They may still call or fold.
* Folding is only offered when facing a bet (checking is free).
* No raising when every opponent is all-in.
* Multiple all-ins, side pots, split pots with odd chips going to the first
  winner clockwise from the button, uncalled bets returned (they form a pot
  only their owner is eligible for).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..poker.evaluator import evaluate_best_codes
from .action import Action, ActionRecord, ActionType, LegalActions
from .pots import Pot, award_pots, build_pots
from .positions import blind_seats, clockwise_from
from .state import BOARD_SIZE, HoldemTableState, SeatState, Street, TableConfig


class IllegalActionError(ValueError):
    """The requested action is not legal in this state."""


# -- construction ----------------------------------------------------------


def start_hand(stacks: Sequence[int], dealer: int, config: TableConfig,
               hole_cards: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
               hand_id: str = "") -> HoldemTableState:
    """Post antes and blinds and return the first decision state."""
    n = len(stacks)
    if not 2 <= n <= 9:
        raise ValueError("2-9 seats supported")
    if not 0 <= dealer < n:
        raise ValueError("dealer out of range")
    if any(int(s) != s or s <= 0 for s in stacks):
        raise ValueError("starting stacks must be positive integers")
    holes = list(hole_cards) if hole_cards is not None else [None] * n
    if len(holes) != n:
        raise ValueError("hole_cards must have one entry per seat")
    seen = set()
    for h in holes:
        if h is None:
            continue
        if len(h) != 2:
            raise ValueError("hole cards come in pairs")
        for c in h:
            if not 0 <= c < 52 or c in seen:
                raise ValueError("invalid or duplicate hole card")
            seen.add(c)

    seats = [SeatState(seat=i, stack=int(stacks[i]),
                       hole_cards=tuple(holes[i]) if holes[i] else None)
             for i in range(n)]
    def post(seat: int, amount: int, street_bet: bool) -> None:
        s = seats[seat]
        paid = min(amount, s.stack)
        seats[seat] = replace(
            s, stack=s.stack - paid,
            committed_total=s.committed_total + paid,
            committed_street=s.committed_street + (paid if street_bet else 0),
            all_in=s.stack - paid == 0)

    if config.ante:
        for i in range(n):
            post(i, config.ante, False)
    sb, bb = blind_seats(n, dealer)
    post(sb, config.small_blind, True)
    post(bb, config.big_blind, True)

    state = HoldemTableState(
        num_seats=n, dealer=dealer, config=config, street=Street.PREFLOP,
        board=(), seats=tuple(seats), actor=None,
        current_bet=config.big_blind, min_raise=config.big_blind,
        last_full_bet=config.big_blind,
        acted_level=(None,) * n, to_act=(),
        hand_id=hand_id)
    first = dealer if n == 2 else (bb + 1) % n
    return _begin_round(state, first)


def _begin_round(state: HoldemTableState, first: int) -> HoldemTableState:
    order = [s for s in clockwise_from(first, state.num_seats)
             if state.seats[s].can_act]
    if len(order) == 1 and state.amount_to_call(order[0]) == 0:
        order = []  # lone player with nothing to call: no betting
    if not order:
        state = replace(state, to_act=(), actor=None)
        return _after_round(state)
    return replace(state, to_act=tuple(order), actor=order[0])


# -- queries -----------------------------------------------------------------


def _others_can_act(state: HoldemTableState, seat: int) -> bool:
    return any(s.can_act for s in state.seats if s.seat != seat)


def _reopened(state: HoldemTableState, seat: int) -> bool:
    level = state.acted_level[seat]
    return level is None or state.current_bet - level >= state.min_raise


def legal_actions(state: HoldemTableState) -> LegalActions:
    if state.actor is None:
        raise IllegalActionError("no player to act")
    seat = state.actor
    s = state.seats[seat]
    to_call = state.amount_to_call(seat)
    max_to = s.committed_street + s.stack
    can_raise = (s.stack > to_call and _reopened(state, seat)
                 and _others_can_act(state, seat))
    min_to = min(state.current_bet + state.min_raise, max_to)
    return LegalActions(
        seat=seat,
        can_fold=to_call > 0,
        can_check=state.current_bet <= s.committed_street,
        call_amount=to_call,
        call_is_all_in=to_call > 0 and to_call == s.stack,
        can_raise=can_raise,
        min_raise_to=min_to if can_raise else 0,
        max_raise_to=max_to if can_raise else 0,
        is_bet=state.current_bet == 0,
    )


def cards_needed(state: HoldemTableState) -> int:
    """Board cards to deal before play can continue (0 if none)."""
    if state.actor is not None or state.street >= Street.RIVER:
        return 0
    if len(state.live_seats) <= 1:
        return 0
    return BOARD_SIZE[Street(state.street + 1)] - len(state.board)


# -- transitions -------------------------------------------------------------


def apply_action(state: HoldemTableState, action: Action) -> HoldemTableState:
    """Apply ``action`` for the current actor and advance the hand."""
    legal = legal_actions(state)
    seat = legal.seat
    s = state.seats[seat]
    t = action.type
    if t == ActionType.ALL_IN:
        if s.stack <= legal.call_amount:
            t, to = ActionType.CALL, None
        elif legal.can_raise:
            t = ActionType.BET if legal.is_bet else ActionType.RAISE
            to = legal.max_raise_to
        else:
            raise IllegalActionError(
                "betting is not reopened: all-in would be a raise; call or fold")
    else:
        to = action.amount

    if t == ActionType.FOLD:
        if not legal.can_fold:
            raise IllegalActionError("cannot fold when checking is free")
        return _record(state, seat, replace(s, folded=True), t, 0, False, False)
    if t == ActionType.CHECK:
        if not legal.can_check:
            raise IllegalActionError("cannot check facing a bet")
        return _record(state, seat, s, t, 0, False, False)
    if t == ActionType.CALL:
        if legal.call_amount == 0:
            raise IllegalActionError("nothing to call; check instead")
        add = legal.call_amount
        new = replace(s, stack=s.stack - add,
                      committed_street=s.committed_street + add,
                      committed_total=s.committed_total + add,
                      all_in=s.stack - add == 0)
        return _record(state, seat, new, t, add, new.all_in, False)
    if t in (ActionType.BET, ActionType.RAISE):
        if not legal.can_raise:
            raise IllegalActionError("raising is not allowed here")
        if (t == ActionType.BET) != legal.is_bet:
            raise IllegalActionError(
                "use BET with no outstanding bet and RAISE otherwise")
        if to is None or int(to) != to:
            raise IllegalActionError("bet/raise needs an integer raise-to amount")
        to = int(to)
        if to > legal.max_raise_to:
            raise IllegalActionError(f"raise-to {to} exceeds stack ({legal.max_raise_to})")
        if to < legal.min_raise_to:
            raise IllegalActionError(
                f"raise-to {to} below minimum {legal.min_raise_to}")
        add = to - s.committed_street
        new = replace(s, stack=s.stack - add, committed_street=to,
                      committed_total=s.committed_total + add,
                      all_in=s.stack - add == 0)
        increment = to - state.current_bet
        full = increment >= state.min_raise
        state = replace(
            state, current_bet=to,
            min_raise=increment if full else state.min_raise,
            last_full_bet=to if full else state.last_full_bet)
        return _record(state, seat, new, t, add, new.all_in, full, raised=True)
    raise IllegalActionError(f"unknown action {action!r}")


def _record(state: HoldemTableState, seat: int, new_seat: SeatState,
            t: ActionType, added: int, all_in: bool, full: bool,
            raised: bool = False) -> HoldemTableState:
    seats = list(state.seats)
    seats[seat] = new_seat
    rec = ActionRecord(street=int(state.street), seat=seat, type=t,
                       added=added, street_total=new_seat.committed_street,
                       all_in=all_in, full_raise=full)
    acted = list(state.acted_level)
    acted[seat] = state.current_bet
    if raised:
        to_act = tuple(x for x in clockwise_from(seat + 1, state.num_seats)
                       if x != seat and seats[x].can_act)
    else:
        to_act = tuple(x for x in state.to_act if x != seat)
    state = replace(state, seats=tuple(seats), acted_level=tuple(acted),
                    to_act=to_act, action_history=state.action_history + (rec,))
    if len(state.live_seats) == 1:
        return settle(replace(state, actor=None, to_act=()))
    if to_act:
        return replace(state, actor=to_act[0])
    return _after_round(replace(state, actor=None))


def _after_round(state: HoldemTableState) -> HoldemTableState:
    """Close the betting round: reset street bets; maybe go to showdown."""
    seats = tuple(replace(s, committed_street=0) for s in state.seats)
    state = replace(state, seats=seats, current_bet=0,
                    min_raise=state.config.big_blind, last_full_bet=0,
                    acted_level=(None,) * state.num_seats, to_act=(),
                    actor=None)
    if state.street == Street.RIVER:
        return replace(state, street=Street.SHOWDOWN)
    return state


def deal_board(state: HoldemTableState, cards: Sequence[int]) -> HoldemTableState:
    """Deal the next street's community cards (explicit, e.g. for replay)."""
    need = cards_needed(state)
    if need == 0:
        raise IllegalActionError("no board cards are due")
    cards = tuple(int(c) for c in cards)
    if len(cards) != need:
        raise IllegalActionError(f"expected {need} board cards, got {len(cards)}")
    if len(set(cards) | set(known_cards(state))) != len(cards) + len(known_cards(state)):
        raise IllegalActionError("duplicate card dealt")
    if any(not 0 <= c < 52 for c in cards):
        raise IllegalActionError("card code out of range")
    street = Street(state.street + 1)
    state = replace(state, street=street, board=state.board + cards)
    first = (state.dealer + 1) % state.num_seats
    return _begin_round(state, first)


def known_cards(state: HoldemTableState) -> Tuple[int, ...]:
    out = list(state.board)
    for s in state.seats:
        if s.hole_cards:
            out.extend(s.hole_cards)
    return tuple(out)


def deal_board_random(state: HoldemTableState,
                      rng: np.random.Generator) -> HoldemTableState:
    """Deal the next street uniformly from cards not yet visible/assigned."""
    dead = set(known_cards(state))
    live = [c for c in range(52) if c not in dead]
    idx = rng.choice(len(live), size=cards_needed(state), replace=False)
    return deal_board(state, [live[i] for i in idx])


def settle(state: HoldemTableState) -> HoldemTableState:
    """Award all pots. With one live player no cards are needed; otherwise the
    board must be complete and every live player's hole cards known."""
    if state.is_complete:
        raise IllegalActionError("hand already settled")
    contributions = [s.committed_total for s in state.seats]
    folded = [s.folded for s in state.seats]
    pots = build_pots(contributions, folded)
    live = state.live_seats
    if len(live) > 1:
        if len(state.board) != 5 or state.street != Street.SHOWDOWN:
            raise IllegalActionError("showdown needs a complete board")
        if any(state.seats[i].hole_cards is None for i in live):
            raise IllegalActionError("showdown needs every live player's cards")
    order = clockwise_from((state.dealer + 1) % state.num_seats, state.num_seats)
    cache: Dict[int, tuple] = {}

    def strength(seat: int) -> tuple:
        if seat not in cache:
            cache[seat] = evaluate_best_codes(
                list(state.seats[seat].hole_cards) + list(state.board))
        return cache[seat]

    won = award_pots(pots, strength, order)
    seats = tuple(replace(s, stack=s.stack + won.get(s.seat, 0),
                          committed_street=0)
                  for s in state.seats)
    return replace(state, seats=seats, street=Street.COMPLETE, actor=None,
                   to_act=(), awards=tuple(sorted(won.items())))


def pots_of(state: HoldemTableState) -> Tuple[Pot, ...]:
    """Current main/side pots (before settlement)."""
    return tuple(build_pots([s.committed_total for s in state.seats],
                            [s.folded for s in state.seats]))


def total_chips(state: HoldemTableState) -> int:
    return sum(s.stack for s in state.seats) + state.pot


def check_invariants(state: HoldemTableState, starting_total: int) -> None:
    """Raise AssertionError if any engine invariant is violated."""
    assert total_chips(state) == starting_total, "chips not conserved"
    assert all(s.stack >= 0 for s in state.seats), "negative stack"
    cards = known_cards(state)
    assert len(cards) == len(set(cards)), "duplicate cards"
    if state.actor is not None:
        actor = state.seats[state.actor]
        assert actor.can_act, "actor cannot act"
        assert state.actor == state.to_act[0]
    if not state.is_complete:
        assert state.pot == sum(s.committed_total for s in state.seats)
        assert max(s.committed_street for s in state.seats) <= \
            max(state.current_bet, 0), "street commitment above current bet"
    else:
        assert sum(w for _, w in state.awards) == \
            sum(s.committed_total for s in state.seats), "awards != pot"
