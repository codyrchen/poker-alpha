"""Native hand-evaluator parity (Phase 10).

The native evaluator packs the Python hand-value tuple into a uint32 that
preserves ordering and equality. These tests pin:

* exhaustive parity on all 2,598,960 five-card hands (slow suite);
* random 6- and 7-card corpora;
* the named edge cases (wheel, board-plays, kickers, ties).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from poker_alpha.poker.cards import card_code
from poker_alpha.poker.evaluator import evaluate_best_codes

from native_helpers import native, pack_hand_value


def _check(cards):
    assert native.debug_evaluate(list(cards)) == \
        pack_hand_value(evaluate_best_codes(list(cards)))


def test_five_card_sample_parity():
    rng = np.random.default_rng(0)
    for _ in range(20_000):
        _check(rng.choice(52, size=5, replace=False).tolist())


@pytest.mark.slow
def test_five_card_exhaustive_parity():
    n = 0
    for hand in itertools.combinations(range(52), 5):
        _check(hand)
        n += 1
    assert n == 2_598_960


def test_six_seven_card_sample_parity():
    rng = np.random.default_rng(1)
    for _ in range(10_000):
        _check(rng.choice(52, size=6, replace=False).tolist())
        _check(rng.choice(52, size=7, replace=False).tolist())


@pytest.mark.slow
def test_seven_card_large_parity():
    rng = np.random.default_rng(2)
    for _ in range(1_000_000):
        _check(rng.choice(52, size=7, replace=False).tolist())


def _codes(*cards):
    return [card_code(c) for c in cards]


def test_named_hands():
    cases = [
        _codes("As", "Ks", "Qs", "Js", "Ts"),             # royal
        _codes("5h", "4h", "3h", "2h", "Ah"),             # wheel SF
        _codes("5h", "4d", "3h", "2h", "Ah"),             # wheel straight
        _codes("Ah", "Ad", "Ac", "As", "Kd"),             # quads
        _codes("Ah", "Ad", "Ac", "Kd", "Ks"),             # boat
        _codes("Ah", "Kh", "9h", "6h", "2h"),             # flush
        _codes("9h", "8d", "7c", "6s", "5h"),             # straight
        _codes("9h", "9d", "9c", "6s", "5h"),             # trips
        _codes("9h", "9d", "6c", "6s", "5h"),             # two pair
        _codes("9h", "9d", "7c", "6s", "5h"),             # pair
        _codes("Kh", "9d", "7c", "6s", "5h"),             # high card
        # 7-card: board plays / kicker battles
        _codes("2c", "3d", "Ah", "Kh", "Qh", "Jh", "Th"), # board royal
        _codes("Ac", "2d", "9h", "9d", "9c", "9s", "Kd"), # quads on board
        _codes("Ac", "Qd", "Ah", "Kd", "7c", "6s", "5h"), # AQ vs board
        _codes("Ac", "Jd", "Ah", "Kd", "7c", "6s", "5h"),
    ]
    for cards in cases:
        _check(cards)


def test_tie_semantics():
    """Equal Python tuples must give equal packed values (split pots)."""
    board = _codes("Ah", "Kd", "Qc", "Js", "9h")
    h0 = _codes("Th", "2c") + board   # same straight both sides
    h1 = _codes("Td", "3c") + board
    assert native.debug_evaluate(h0) == native.debug_evaluate(h1)
    assert evaluate_best_codes(h0) == evaluate_best_codes(h1)


def test_packing_is_order_isomorphic():
    """Sorting by Python tuple and by packed value must agree."""
    rng = np.random.default_rng(3)
    hands = [rng.choice(52, size=7, replace=False).tolist() for _ in range(5_000)]
    tuples = [evaluate_best_codes(h) for h in hands]
    packed = [pack_hand_value(t) for t in tuples]
    order_t = sorted(range(len(hands)), key=lambda i: (tuples[i], i))
    order_p = sorted(range(len(hands)), key=lambda i: (packed[i], i))
    assert order_t == order_p
