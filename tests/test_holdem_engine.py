"""Phase 8: multiplayer no-limit Hold'em rules engine scenarios."""

import numpy as np
import pytest

from poker_alpha.holdem import (Action, ActionType, IllegalActionError,
                                Street, TableConfig, apply_action,
                                cards_needed, check_invariants, deal_board,
                                deal_board_random, legal_actions,
                                position_names, pots_of, settle, start_hand,
                                total_chips)
from poker_alpha.holdem.pots import Pot
from poker_alpha.poker import card_code

CFG = TableConfig(small_blind=1, big_blind=2)


def C(*cards):
    return [card_code(c) for c in cards]


def holes(*pairs):
    return [tuple(C(*p.split())) for p in pairs]


def act(state, *actions):
    total = total_chips(state)
    for a in actions:
        state = apply_action(state, a)
        check_invariants(state, total)
    return state


def run_board(state, *streets):
    """Deal given streets while cards are due, then settle at showdown."""
    total = total_chips(state)
    for cards in streets:
        assert state.actor is None and cards_needed(state) == len(cards.split())
        state = deal_board(state, C(*cards.split()))
        check_invariants(state, total)
    return state


# -- positions & blinds --------------------------------------------------------

def test_position_names():
    assert position_names(2, 0) == {0: "BTN", 1: "BB"}
    assert position_names(6, 0) == {0: "BTN", 1: "SB", 2: "BB", 3: "UTG",
                                     4: "HJ", 5: "CO"}
    names9 = position_names(9, 4)
    assert names9[4] == "BTN" and names9[5] == "SB" and names9[6] == "BB"
    assert names9[7] == "UTG" and names9[3] == "CO"
    assert len(set(names9.values())) == 9


def test_heads_up_dealer_posts_sb_acts_first_preflop_last_postflop():
    s = start_hand([100, 100], dealer=1, config=CFG)
    assert s.seats[1].committed_street == 1 and s.seats[0].committed_street == 2
    assert s.actor == 1                      # button/SB first preflop
    s = act(s, Action.call())
    assert s.actor == 0                      # BB option
    la = legal_actions(s)
    assert la.can_check and not la.can_fold and la.can_raise
    s = act(s, Action.check())
    assert s.actor is None and cards_needed(s) == 3
    s = deal_board(s, C("2c", "7d", "9h"))
    assert s.street == Street.FLOP and s.actor == 0   # BB first postflop


def test_six_max_preflop_and_postflop_order_with_antes():
    cfg = TableConfig(small_blind=1, big_blind=2, ante=1)
    s = start_hand([200] * 6, dealer=0, config=cfg)
    assert s.pot == 6 + 3
    assert s.seats[1].committed_total == 2 and s.seats[1].committed_street == 1
    assert s.actor == 3                      # UTG
    s = act(s, Action.fold(), Action.fold(), Action.call(),  # UTG, HJ fold; CO calls
            Action.call(),                   # BTN
            Action.call(),                   # SB completes
            Action.check())                  # BB
    s = deal_board(s, C("2c", "7d", "9h"))
    assert s.actor == 1                      # SB first live seat left of BTN


def test_min_bet_and_min_raise_rules():
    s = start_hand([100, 100, 100], dealer=0, config=CFG)
    la = legal_actions(s)                    # BTN (UTG 3-handed) faces 2
    assert la.min_raise_to == 4 and la.max_raise_to == 100
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.raise_to(3))
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.bet(6))       # must use RAISE preflop
    s = act(s, Action.raise_to(7))           # raise increment 5
    assert legal_actions(s).min_raise_to == 12
    s = act(s, Action.raise_to(20))          # SB re-raises, increment 13
    assert legal_actions(s).min_raise_to == 33
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.check())


def test_fold_not_allowed_when_check_is_free():
    s = start_hand([100, 100], dealer=0, config=CFG)
    s = act(s, Action.call())
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.fold())


def test_everyone_folds_uncalled_bet_returned():
    s = start_hand([100, 100, 100], dealer=0, config=CFG)
    s = act(s, Action.raise_to(10), Action.fold(), Action.fold())
    assert s.is_complete
    assert dict(s.awards) == {0: 13}
    assert [x.stack for x in s.seats] == [103, 99, 98]


def test_overbet_and_call_to_showdown_heads_up():
    s = start_hand([100, 100], dealer=0, config=CFG,
                   hole_cards=holes("As Ah", "Ks Kd"))
    s = act(s, Action.call(), Action.check())
    s = run_board(s, "2c 7d 9h")
    s = act(s, Action.check(), Action.bet(30))   # BTN overbets 30 into 4
    s = act(s, Action.call())
    s = run_board(s, "3c")
    s = act(s, Action.check(), Action.check())
    s = run_board(s, "4d")
    s = act(s, Action.check(), Action.check())
    assert s.street == Street.SHOWDOWN
    s = settle(s)
    assert dict(s.awards) == {0: 64}
    assert [x.stack for x in s.seats] == [132, 68]


def test_all_in_call_runs_out_board_without_betting():
    s = start_hand([50, 100], dealer=0, config=CFG,
                   hole_cards=holes("As Ah", "Ks Kd"))
    s = act(s, Action.all_in())
    assert legal_actions(s).can_raise is False   # opponent all-in
    s = act(s, Action.call())
    assert s.seats[1].committed_total == 50      # only matches the 50
    s = run_board(s, "2c 7d 9h", "3c", "Kh")
    assert s.street == Street.SHOWDOWN
    s = settle(s)
    assert [x.stack for x in s.seats] == [0, 150]   # set of kings wins


def test_short_stack_blind_all_in():
    s = start_hand([1, 100, 100], dealer=0, config=TableConfig(1, 2))
    # 3-handed: seat 1 SB, seat 2 BB; seat 0 ok. Make the BB short instead:
    s = start_hand([100, 100, 1], dealer=0, config=CFG)
    assert s.seats[2].all_in and s.seats[2].committed_total == 1
    assert s.current_bet == 2                    # others still owe a full BB
    assert s.actor == 0


def test_short_all_in_does_not_reopen_betting():
    # Seat order 3-handed, dealer 0: UTG=0 (BTN), SB=1, BB=2.
    s = start_hand([200, 200, 25], dealer=0, config=CFG)
    s = act(s, Action.raise_to(10))      # BTN raises to 10 (increment 8)
    s = act(s, Action.call())            # SB calls 10
    s = act(s, Action.all_in())          # BB shoves 25: increment 15 >= 8, full
    assert legal_actions(s).can_raise    # full raise reopens for BTN

    s = start_hand([200, 200, 15], dealer=0, config=CFG)
    s = act(s, Action.raise_to(10), Action.call())
    s = act(s, Action.all_in())          # BB shoves 15: increment 5 < 8, short
    la = legal_actions(s)
    assert s.actor == 0 and la.call_amount == 5
    assert not la.can_raise              # BTN already acted: not reopened
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.all_in())
    s = act(s, Action.call())
    assert s.actor == 1 and not legal_actions(s).can_raise
    s = act(s, Action.call())
    assert s.actor is None and cards_needed(s) == 3


@pytest.mark.parametrize("bb_stack,reopened", [(21, False), (23, True)])
def test_cumulative_short_all_ins_reopen(bb_stack, reopened):
    # seats: 0 BTN, 1 SB, 2 BB, 3 UTG. Limped pot, then UTG bets 10 on the
    # flop, BTN calls, SB shoves short (to 14), BB shoves to bb_stack - 2.
    s = start_hand([200, 16, bb_stack, 200], dealer=0, config=CFG)
    s = act(s, Action.call(), Action.call(), Action.call(), Action.check())
    s = deal_board(s, C("2c", "7d", "9h"))
    assert s.actor == 1
    s = act(s, Action.check(), Action.check(), Action.bet(10), Action.call())
    s = act(s, Action.all_in())           # SB to 14: increment 4 < 10
    s = act(s, Action.all_in())           # BB to 19 (short) or 21 (cumulative 11 >= 10)
    assert s.actor == 3
    assert legal_actions(s).can_raise is reopened


def test_multiway_all_ins_side_pots_and_split():
    hc = holes("As Ks", "Ad Kd", "7c 7d", "2h 3h")
    s = start_hand([100, 40, 70, 100], dealer=0, config=CFG, hole_cards=hc)
    # UTG is seat 3.
    s = act(s, Action.all_in(), Action.call(), Action.all_in(), Action.all_in())
    assert s.actor is None
    pots = pots_of(s)
    assert pots == (Pot(160, (0, 1, 2, 3)), Pot(90, (0, 2, 3)), Pot(60, (0, 3)))
    s = run_board(s, "Ac Kc 2d", "9s", "8s")
    s = settle(s)
    # Seats 0 and 1 tie with AK two pair; seat 1 only eligible for the main.
    won = dict(s.awards)
    assert won == {1: 80, 0: 80 + 90 + 60}
    assert sum(x.stack for x in s.seats) == 310


def test_one_player_covering_everyone_nine_handed():
    stacks = [500, 20, 30, 40, 50, 60, 70, 80, 90]
    s = start_hand(stacks, dealer=0, config=CFG)
    total = sum(stacks)
    # UTG = seat 3 ... everyone shoves, seat 0 calls last.
    while s.actor is not None:
        s = apply_action(s, Action.call() if s.actor == 0 else Action.all_in())
        check_invariants(s, total)
    pots = pots_of(s)
    assert len(pots) == 8
    assert pots[-1].eligible == (0, 8)
    rng = np.random.default_rng(0)
    # No hole cards known -> showdown must refuse.
    while cards_needed(s):
        s = deal_board_random(s, rng)
    with pytest.raises(IllegalActionError):
        settle(s)


def test_dead_folded_contributions_stay_in_pot():
    hc = holes("As Ks", "Qd Qc", "7c 2d")
    s = start_hand([100, 100, 100], dealer=0, config=CFG, hole_cards=hc)
    s = act(s, Action.raise_to(6), Action.call(), Action.raise_to(20),
            Action.call(), Action.fold())
    assert s.seats[1].folded and s.seats[1].committed_total == 6
    s = run_board(s, "2c 2h 9d")
    s = act(s, Action.check(), Action.check())
    s = run_board(s, "3s")
    s = act(s, Action.check(), Action.check())
    s = run_board(s, "4s")
    s = act(s, Action.check(), Action.check())
    s = settle(s)
    assert dict(s.awards) == {2: 46}       # 7-2 makes trips, wins dead money too


def test_illegal_requests_rejected():
    s = start_hand([100, 100], dealer=0, config=CFG)
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.raise_to(1000))
    with pytest.raises(IllegalActionError):
        apply_action(s, Action.check())
    with pytest.raises(ValueError):
        start_hand([100], 0, CFG)
    with pytest.raises(ValueError):
        start_hand([100, 100], 0, CFG, hole_cards=holes("As Ks", "As Kd"))
    s = act(s, Action.call(), Action.check())
    with pytest.raises(IllegalActionError):
        deal_board(s, C("2c", "7d"))
    with pytest.raises(IllegalActionError):
        settle(s)


def _random_hand(rng, n):
    stacks = [int(x) for x in rng.integers(5, 400, size=n)]
    deck = rng.permutation(52)
    hc = [(int(deck[2 * i]), int(deck[2 * i + 1])) for i in range(n)]
    cfg = TableConfig(1, 2, ante=int(rng.integers(0, 2)))
    s = start_hand(stacks, int(rng.integers(n)), cfg, hole_cards=hc)
    total = sum(stacks)
    steps = 0
    while not s.is_complete:
        steps += 1
        if s.actor is not None:
            la = legal_actions(s)
            options = []
            if la.can_fold:
                options.append(Action.fold())
            options.append(Action.check() if la.can_check else Action.call())
            if la.can_raise:
                lo, hi = la.min_raise_to, la.max_raise_to
                to = int(rng.integers(lo, hi + 1))
                options.append(Action.bet(to) if la.is_bet else Action.raise_to(to))
                options.append(Action.all_in())
            s = apply_action(s, options[int(rng.integers(len(options)))])
        elif cards_needed(s):
            s = deal_board_random(s, rng)
        else:
            s = settle(s)
        check_invariants(s, total)
        assert steps < 500
    assert sum(x.stack for x in s.seats) == total
    return s


@pytest.mark.parametrize("n", [2, 3, 6, 9])
def test_random_play_conserves_chips(n):
    rng = np.random.default_rng(100 + n)
    for _ in range(150):
        _random_hand(rng, n)
