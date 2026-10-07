"""Phase 8: side-pot construction and awarding, tested independently."""

import pytest

from poker_alpha.holdem.pots import Pot, award_pots, build_pots


def test_single_pot_everyone_equal():
    assert build_pots([10, 10, 10], [False] * 3) == [Pot(30, (0, 1, 2))]


def test_one_short_all_in_creates_side_pot():
    # Seat 0 all-in for 5, seats 1,2 put in 20 each.
    assert build_pots([5, 20, 20], [False] * 3) == [
        Pot(15, (0, 1, 2)), Pot(30, (1, 2))]


def test_multiple_side_pots():
    pots = build_pots([10, 25, 50, 50], [False] * 4)
    assert pots == [Pot(40, (0, 1, 2, 3)), Pot(45, (1, 2, 3)), Pot(50, (2, 3))]
    assert sum(p.amount for p in pots) == 135


def test_folded_contributions_are_dead_money():
    # Seat 1 put in 30 and folded; seat 0 all-in 10; seat 2 put in 40.
    pots = build_pots([10, 30, 40], [False, True, False])
    assert pots == [Pot(30, (0, 2)), Pot(50, (2,))]
    assert sum(p.amount for p in pots) == 80


def test_uncalled_bet_forms_owner_only_pot():
    pots = build_pots([20, 100], [False, False])
    assert pots == [Pot(40, (0, 1)), Pot(80, (1,))]


def test_one_player_covering_everyone():
    pots = build_pots([100, 30, 60, 10], [False] * 4)
    assert pots[0] == Pot(40, (0, 1, 2, 3))
    assert pots[1] == Pot(60, (0, 1, 2))
    assert pots[2] == Pot(60, (0, 2))
    assert pots[3] == Pot(40, (0,))


def test_identical_eligibility_layers_merge():
    # Folded seat at an intermediate level doesn't split the live pot.
    assert build_pots([20, 10, 20], [False, True, False]) == [Pot(50, (0, 2))]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        build_pots([1.5, 2], [False, False])
    with pytest.raises(ValueError):
        build_pots([1, 2], [False])


def _strength(values):
    return lambda seat: values[seat]


def test_award_best_hand_takes_pot():
    won = award_pots([Pot(30, (0, 1, 2))], _strength({0: (1,), 1: (5,), 2: (3,)}),
                     [0, 1, 2])
    assert won == {1: 30}


def test_tied_main_pot_splits_with_odd_chip_order():
    won = award_pots([Pot(31, (0, 1, 2))], _strength({0: (5,), 1: (5,), 2: (1,)}),
                     seat_order=[1, 2, 0])
    assert won == {1: 16, 0: 15}


def test_short_stack_wins_main_bigger_wins_side():
    pots = build_pots([5, 20, 20], [False] * 3)
    won = award_pots(pots, _strength({0: (9,), 1: (5,), 2: (1,)}), [0, 1, 2])
    assert won == {0: 15, 1: 30}


def test_tied_side_pot():
    pots = build_pots([5, 20, 20], [False] * 3)
    won = award_pots(pots, _strength({0: (1,), 1: (5,), 2: (5,)}), [0, 1, 2])
    assert won == {1: 8 + 15, 2: 7 + 15}  # odd chip: seat 1 first in order
    assert sum(won.values()) == 45
