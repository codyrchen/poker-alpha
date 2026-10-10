"""Phase 6: configurable betting abstraction, separate from core rules."""

import pytest

from poker_alpha.abstraction.betting import (ActionAbstraction,
                                             BettingContext)

FULL = ActionAbstraction(
    bet_fractions=(0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0),
    raise_fractions=(0.5, 0.75, 1.0, 1.5, 2.0),
)


def amounts(abs_, ctx):
    return {label: (a.kind, round(a.raise_to, 6), round(a.add, 6))
            for label, a in abs_.menu(ctx)}


def test_signature_is_versioned_and_config_aware():
    a = ActionAbstraction()
    assert a.signature() == \
        "ActionAbstraction:v1:bets=0.33,0.75,1:raises=0.75,1:allin=1"
    assert FULL.signature() != a.signature()


def test_deep_stack_unopened_offers_every_bet_size():
    ctx = BettingContext(pot=10, to_call=0, hero_stack=200,
                         max_opponent_stack=200)
    m = amounts(FULL, ctx)
    assert list(m) == ["check", "bet_25", "bet_33", "bet_50", "bet_75",
                       "bet_100", "bet_150", "bet_200", "all_in"]
    assert m["bet_75"] == ("bet", 7.5, 7.5)
    assert m["bet_200"] == ("bet", 20.0, 20.0)
    assert m["all_in"] == ("all_in", 200.0, 200.0)
    assert "fold" not in m  # nothing to fold to


def test_tiny_pot_lifts_small_bets_to_min_bet_and_dedupes():
    # Limped pot of 2 BB: 25%/33%/50% of pot are all <= 1 BB min bet.
    ctx = BettingContext(pot=2, to_call=0, hero_stack=99,
                         max_opponent_stack=99, big_blind=1,
                         min_raise_increment=1)
    m = amounts(FULL, ctx)
    sizes = [v[1] for k, v in m.items() if k.startswith("bet_")]
    assert len(sizes) == len(set(sizes))           # no two labels, one wager
    assert min(sizes) == pytest.approx(1.0)        # min bet = 1 BB
    assert "bet_25" in m and "bet_33" not in m and "bet_50" not in m


def test_large_pot_with_short_stack_collapses_to_all_in():
    ctx = BettingContext(pot=40, to_call=0, hero_stack=12,
                         max_opponent_stack=80)
    m = amounts(FULL, ctx)
    assert list(m) == ["check", "bet_25", "all_in"]
    assert m["all_in"] == ("all_in", 12.0, 12.0)
    for label, (_, to, _) in m.items():
        assert to <= 12.0 + 1e-9
    # 25% of 40 = 10 < 12 is still a real (non all-in) bet.
    assert m["bet_25"] == ("bet", 10.0, 10.0)


def test_short_stack_facing_bet_can_only_call_all_in_or_fold():
    ctx = BettingContext(pot=30, to_call=10, hero_stack=6,
                         max_opponent_stack=50)
    m = amounts(FULL, ctx)
    assert list(m) == ["fold", "call"]
    assert m["call"] == ("all_in", 6.0, 6.0)  # all-in for less


def test_facing_bet_uses_pot_relative_raises_and_min_raise():
    # Pot 10 before a 5 bet -> pot 15, hero to call 5.
    ctx = BettingContext(pot=15, to_call=5, hero_stack=100,
                         max_opponent_stack=100, min_raise_increment=5)
    m = amounts(FULL, ctx)
    assert list(m)[:2] == ["fold", "call"]
    assert m["call"] == ("call", 5.0, 5.0)
    # raise_100: call 5, then add pot-after-call 20 -> raise to 25.
    assert m["raise_100"] == ("raise", 25.0, 25.0)
    # raise_50 -> 5 + 10 = 15 >= min raise-to 10.
    assert m["raise_50"] == ("raise", 15.0, 15.0)
    for label, (kind, to, _) in m.items():
        if kind == "raise":
            assert to >= ctx.min_raise_to - 1e-9


def test_facing_raise_min_raise_lift():
    # Hero bet 10, villain raised to 40: to_call 30, last increment 30.
    ctx = BettingContext(pot=60, to_call=30, hero_stack=200, hero_street=10,
                         max_opponent_stack=160, min_raise_increment=30)
    abs_ = ActionAbstraction(bet_fractions=(), raise_fractions=(0.1, 1.0))
    m = amounts(abs_, ctx)
    # 10% raise would be to 49 < min raise-to 70: lifted to 70.
    assert m["raise_10"] == ("raise", 70.0, 60.0)
    assert m["raise_100"] == ("raise", 40 + 90.0, 120.0)


def test_cannot_raise_all_in_opponent_and_cap_at_callable_amount():
    ctx = BettingContext(pot=50, to_call=20, hero_stack=200,
                         max_opponent_stack=0)
    assert ActionAbstraction().legal_actions(ctx) == ["fold", "call"]
    # Opponent has 30 behind: hero's "all in" is capped at what can be called.
    ctx2 = BettingContext(pot=20, to_call=0, hero_stack=500,
                          max_opponent_stack=30)
    m = amounts(FULL, ctx2)
    assert m["all_in"] == ("bet", 30.0, 30.0)
    assert all(v[1] <= 30.0 + 1e-9 for v in m.values())


def test_to_concrete_and_illegal_label():
    ctx = BettingContext(pot=10, to_call=0, hero_stack=100,
                         max_opponent_stack=100)
    assert ActionAbstraction().to_concrete("bet_75", ctx).raise_to == 7.5
    with pytest.raises(ValueError):
        ActionAbstraction().to_concrete("raise_75", ctx)


def test_translate_pseudo_harmonic_mapping():
    abs_ = ActionAbstraction(bet_fractions=(0.5, 1.0), raise_fractions=())
    ctx = BettingContext(pot=10, to_call=0, hero_stack=100,
                         max_opponent_stack=100)
    assert abs_.translate("bet", 5.0, ctx) == {"bet_50": 1.0}
    assert abs_.translate("bet", 10.0, ctx) == {"bet_100": 1.0}
    probs = abs_.translate("bet", 7.5, ctx)
    # f(x) = (B - x)(1 + A) / ((B - A)(1 + x)) with A=.5, B=1, x=.75
    expected = (1 - 0.75) * 1.5 / (0.5 * 1.75)
    assert probs["bet_50"] == pytest.approx(expected)
    assert probs["bet_100"] == pytest.approx(1 - expected)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert abs_.nearest("bet", 6.0, ctx) == "bet_50"
    assert abs_.nearest("bet", 9.5, ctx) == "bet_100"
    assert abs_.translate("bet", 1.0, ctx) == {"bet_50": 1.0}   # below menu
    assert abs_.translate("bet", 100.0, ctx) == {"all_in": 1.0}
    assert abs_.translate("fold", 0, BettingContext(10, 5, 50, 50)) == {"fold": 1.0}
    assert abs_.translate("call", 0, BettingContext(10, 5, 50, 50)) == {"call": 1.0}
    assert abs_.translate("check", 0, ctx) == {"check": 1.0}


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        BettingContext(pot=-1, to_call=0, hero_stack=1, max_opponent_stack=1)
