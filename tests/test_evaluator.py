"""Tests for the Hold'em hand evaluator: every category, ties, and ordering."""

import pytest

from poker_alpha.poker import compare_hands, evaluate_best, evaluate_five, codes
from poker_alpha.poker.evaluator import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
)


def value(*cards: str):
    return evaluate_five(codes(cards))


# -- categories --------------------------------------------------------------

def test_high_card():
    v = value("As", "Kd", "9h", "5c", "2s")
    assert v[0] == HIGH_CARD
    assert v[1:] == (12, 11, 7, 3, 0)


def test_one_pair():
    v = value("As", "Ad", "9h", "5c", "2s")
    assert v[0] == ONE_PAIR
    assert v[1] == 12          # pair of aces
    assert v[2:] == (7, 3, 0)  # kickers descending


def test_two_pair():
    v = value("As", "Ad", "9h", "9c", "2s")
    assert v[0] == TWO_PAIR
    assert v[1:] == (12, 7, 0)  # aces, nines, deuce kicker


def test_three_of_a_kind():
    v = value("As", "Ad", "Ah", "9c", "2s")
    assert v[0] == THREE_OF_A_KIND
    assert v[1:] == (12, 7, 0)


def test_straight_and_wheel():
    v = value("9s", "8d", "7h", "6c", "5s")
    assert v == (STRAIGHT, 7)  # nine-high straight (rank 7)
    wheel = value("As", "2d", "3h", "4c", "5s")
    assert wheel == (STRAIGHT, 3)  # five-high straight
    # The wheel is the LOWEST straight.
    assert value("6s", "5d", "4h", "3c", "2s") > wheel


def test_flush():
    v = value("As", "Js", "9s", "5s", "2s")
    assert v[0] == FLUSH
    assert v[1:] == (12, 9, 7, 3, 0)


def test_full_house():
    v = value("As", "Ad", "Ah", "9c", "9s")
    assert v == (FULL_HOUSE, 12, 7)
    # Trips rank dominates: KKK-AA beats ... no wait, AAA-99 vs KKK-AA
    assert v > value("Ks", "Kd", "Kh", "Ac", "As")


def test_four_of_a_kind():
    v = value("As", "Ad", "Ah", "Ac", "9s")
    assert v == (FOUR_OF_A_KIND, 12, 7)
    assert v > value("Ks", "Kd", "Kh", "Kc", "As")


def test_straight_flush_and_royal():
    v = value("9s", "8s", "7s", "6s", "5s")
    assert v == (STRAIGHT_FLUSH, 7)
    royal = value("As", "Ks", "Qs", "Js", "Ts")
    assert royal == (STRAIGHT_FLUSH, 12)
    steel_wheel = value("As", "2s", "3s", "4s", "5s")
    assert steel_wheel == (STRAIGHT_FLUSH, 3)
    assert royal > v > steel_wheel


def test_category_ordering_is_total():
    ladder = [
        value("As", "Kd", "9h", "5c", "2s"),          # high card
        value("As", "Ad", "9h", "5c", "2s"),          # pair
        value("As", "Ad", "9h", "9c", "2s"),          # two pair
        value("As", "Ad", "Ah", "9c", "2s"),          # trips
        value("9s", "8d", "7h", "6c", "5s"),          # straight
        value("As", "Js", "9s", "5s", "2s"),          # flush
        value("As", "Ad", "Ah", "9c", "9s"),          # full house
        value("As", "Ad", "Ah", "Ac", "9s"),          # quads
        value("9s", "8s", "7s", "6s", "5s"),          # straight flush
    ]
    for weaker, stronger in zip(ladder, ladder[1:]):
        assert stronger > weaker


# -- ties --------------------------------------------------------------------

def test_identical_ranks_tie_across_suits():
    a = value("As", "Kd", "9h", "5c", "2s")
    b = value("Ah", "Kc", "9d", "5s", "2h")
    assert a == b


def test_kicker_breaks_tie():
    assert value("As", "Ad", "Kh", "5c", "2s") > value("Ah", "Ac", "Qh", "5d", "2d")


def test_board_plays_tie_on_seven_cards():
    board = ["As", "Ks", "Qs", "Js", "Ts"]  # royal flush on the board
    assert compare_hands(["2c", "3d"] + board, ["9h", "9d"] + board) == 0


# -- best-of-seven -----------------------------------------------------------

def test_evaluate_best_finds_hidden_straight():
    seven = ["9s", "8d", "7h", "6c", "5s", "Ah", "Ad"]
    assert evaluate_best(seven)[0] == STRAIGHT


def test_evaluate_best_prefers_flush_over_straight():
    seven = ["9s", "8s", "7s", "6c", "5s", "2s", "Ad"]
    assert evaluate_best(seven)[0] == FLUSH


def test_compare_hands_hero_wins_and_loses():
    board = ["2h", "7d", "9c", "Jd", "3s"]
    assert compare_hands(["As", "Ad"] + board, ["Ks", "Kd"] + board) == 1
    assert compare_hands(["Ks", "Kd"] + board, ["As", "Ad"] + board) == -1


def test_evaluate_best_rejects_wrong_sizes():
    with pytest.raises(ValueError):
        evaluate_best(["As", "Kd"])
    with pytest.raises(ValueError):
        evaluate_best(["As", "Kd", "9h", "5c", "2s", "3d", "4h", "6s"])
