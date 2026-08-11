"""Tests for Card / Deck / Hand primitives."""

import numpy as np
import pytest

from poker_alpha.poker import Card, Deck, Hand, card_code, card_str, codes


def test_card_roundtrip_all_52():
    seen = set()
    for code in range(52):
        s = card_str(code)
        assert card_code(s) == code
        seen.add(s)
    assert len(seen) == 52


def test_card_rank_and_suit():
    aspades = Card.from_str("As")
    assert aspades.rank == 12
    assert aspades.suit == 3
    deuce_clubs = Card.from_str("2c")
    assert deuce_clubs.rank == 0
    assert deuce_clubs.suit == 0
    assert str(aspades) == "As"


def test_card_rejects_bad_input():
    with pytest.raises(ValueError):
        Card.from_str("Xx")
    with pytest.raises(ValueError):
        Card(52)
    with pytest.raises(ValueError):
        Card(-1)


def test_codes_normalizes_mixed_inputs():
    assert codes(["As", Card.from_str("Kd"), 0]) == [51, 24, 0]
    with pytest.raises(TypeError):
        codes([1.5])
    with pytest.raises(ValueError):
        codes([99])


def test_deck_draw_and_exclude():
    deck = Deck(rng=np.random.default_rng(0), exclude=["As", "Ah"])
    assert len(deck) == 50
    drawn = deck.draw(5)
    assert len(drawn) == 5
    assert len(deck) == 45
    assert card_code("As") not in drawn


def test_deck_shuffle_is_seeded():
    d1 = Deck(rng=np.random.default_rng(7))
    d1.shuffle()
    d2 = Deck(rng=np.random.default_rng(7))
    d2.shuffle()
    assert d1.cards == d2.cards


def test_deck_overdraw_raises():
    deck = Deck(rng=np.random.default_rng(0))
    deck.draw(52)
    with pytest.raises(ValueError):
        deck.draw(1)


def test_hand_from_strs():
    hand = Hand.from_strs("As", "Kd")
    assert len(hand) == 2
    assert str(hand) == "As Kd"
