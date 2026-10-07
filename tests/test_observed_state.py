"""Phase 9: normalized hero-visible state, validation and adapters."""

import json
from dataclasses import replace

import numpy as np
import pytest

from poker_alpha.holdem import (Action, ManualStateAdapter, ObservedSeat,
                                SimulationStateAdapter, Street, TableConfig,
                                apply_action, deal_board_random, is_valid,
                                observed_to_dict, start_hand, validate)
from poker_alpha.poker import card_code


def base_dict(**overrides):
    d = {
        "num_seats": 6, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "As Ks", "board": "Qs Js 4h", "pot": 13.5,
        "actor": 0,
        "seats": [
            {"stack": 96.5, "bet": 0}, {"stack": 0, "occupied": False},
            {"stack": 90.0, "bet": 0}, {"stack": 50, "folded": True},
            {"stack": 120.0, "bet": 0}, {"stack": 80.0, "folded": True},
        ],
    }
    d.update(overrides)
    return d


def codes_of(kinds):
    return {i.code for i in kinds}


def test_manual_adapter_parses_and_derives():
    s = ManualStateAdapter.from_dict(base_dict())
    assert s.street == Street.FLOP                  # inferred from board
    assert s.hero_cards == (card_code("As"), card_code("Ks"))
    assert s.opponents_in_hand == (2, 4)
    assert s.effective_stack == 96.5
    assert s.amount_to_call == 0 and s.pot_odds is None
    assert s.spr == pytest.approx(96.5 / 13.5)
    assert s.hero_position == "BTN"
    assert is_valid(s), validate(s)


def test_pot_odds_when_facing_a_bet():
    d = base_dict(actor=0, pot=23.5)
    d["seats"][2]["bet"] = 10.0
    s = ManualStateAdapter.from_dict(d)
    assert s.amount_to_call == 10.0
    assert s.pot_odds == pytest.approx(10 / 33.5)


def test_round_trip_through_json():
    s = ManualStateAdapter.from_dict(base_dict())
    again = ManualStateAdapter.from_dict(json.loads(json.dumps(observed_to_dict(s))))
    assert again == s
    with pytest.raises(ValueError):
        ManualStateAdapter.from_dict({**observed_to_dict(s), "format": "x/v9"})


@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.update(board="Qs Qs 4h"), "duplicate_cards"),
    (lambda d: d.update(board="As Js 4h"), "hero_board_collision"),
    (lambda d: d.update(board="Qs Js"), "board_size"),
    (lambda d: d.update(street="TURN"), "board_street"),
    (lambda d: d["seats"][2].update(stack=-5), "negative_stack"),
    (lambda d: d["seats"][2].update(bet=20), "pot_below_bets"),
    (lambda d: d.update(actor=3), "actor"),
    (lambda d: d.update(actor=1), "actor"),
    (lambda d: d.update(hero_seat=1), "hero_seat"),
    (lambda d: d.update(dealer=1), "dealer"),
    (lambda d: d["seats"][2].update(all_in=True), "all_in_with_chips"),
])
def test_validation_catches_impossible_states(mutate, code):
    d = base_dict()
    mutate(d)
    s = ManualStateAdapter.from_dict(d)
    assert code in codes_of(validate(s))
    assert not is_valid(s)


def test_pot_mismatch_with_known_contributions():
    d = base_dict(pot=20.0)
    for seat, c in zip(d["seats"], [5, 0, 5, 1, 5, 0]):
        seat["committed"] = c
    assert "pot_mismatch" in codes_of(validate(ManualStateAdapter.from_dict(d)))
    d["pot"] = 16.0
    assert is_valid(ManualStateAdapter.from_dict(d))


def test_opponent_cards_not_required_and_not_leaked():
    hc = [(card_code("As"), card_code("Ks")), (card_code("2c"), card_code("2d")),
          (card_code("9h"), card_code("8h"))]
    eng = start_hand([200, 200, 200], dealer=0, config=TableConfig(1, 2),
                     hole_cards=hc, hand_id="h1")
    eng = apply_action(eng, Action.raise_to(6))
    obs = SimulationStateAdapter.observe(eng, hero_seat=1)
    assert obs.hero_cards == hc[1]
    flat = json.dumps(observed_to_dict(obs))
    assert "As" not in flat and "9h" not in flat
    assert obs.pot == 9 and obs.amount_to_call == 5 and obs.actor == 1
    assert obs.action_history[-1].kind == "raise"
    assert obs.action_history[-1].amount == 6
    assert is_valid(obs), validate(obs)


def test_simulation_adapter_valid_through_random_hands():
    rng = np.random.default_rng(3)
    for _ in range(30):
        deck = rng.permutation(52)
        hc = [(int(deck[2 * i]), int(deck[2 * i + 1])) for i in range(6)]
        s = start_hand([100] * 6, 0, TableConfig(1, 2), hole_cards=hc)
        while not s.is_complete:
            if s.actor is not None:
                obs = SimulationStateAdapter.observe(s, s.actor)
                errors = [i for i in validate(obs) if i.severity == "error"]
                assert not errors, errors
                s = apply_action(s, Action.call() if obs.amount_to_call else Action.check())
            elif len(s.board) < 5 and s.street < Street.RIVER:
                s = deal_board_random(s, rng)
            else:
                break
