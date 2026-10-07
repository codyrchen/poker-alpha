"""Phase 10: combo-level weighted ranges, priors and Bayesian updates."""

import math
from types import SimpleNamespace as A

import numpy as np
import pytest

from poker_alpha.opponent import (ARCHETYPE_MODELS, BehaviorModel,
                                  RangeBelief, RangePriors,
                                  StrategyLikelihood, classify_preflop_line,
                                  update_range_for_action)
from poker_alpha.poker import card_code
from poker_alpha.poker.ranges import (COMBOS, NUM_COMBOS, WeightedRange,
                                      blocked_mask, combo_index, parse_range,
                                      strength_percentiles)


def C(*cards):
    return [card_code(c) for c in cards]


def test_combo_table():
    assert NUM_COMBOS == 1326
    assert len({tuple(c) for c in COMBOS}) == 1326
    assert combo_index(["Ks", "As"]) == combo_index(["As", "Ks"])


def test_blocker_math_is_exact():
    u = WeightedRange.uniform()
    assert u.num_live_combos == 1326
    hero = u.remove_cards(C("As", "Ks"))
    assert hero.num_live_combos == math.comb(50, 2) == 1225
    flop = hero.condition_on_board(C("Qs", "Js", "4h"))
    assert flop.num_live_combos == math.comb(47, 2) == 1081
    # Class masses after blockers: AA has C(3,2)=3 combos left of 1225.
    assert hero.class_mass("AA") == pytest.approx(3 / 1225)
    assert hero.class_mass("AKs") == pytest.approx(3 / 1225)
    assert hero.class_mass("AKo") == pytest.approx(6 / 1225)  # 3 aces x 3 kings - 3 suited
    assert hero.class_mass("QQ") == pytest.approx(6 / 1225)
    assert flop.class_mass("QQ") == pytest.approx(3 / 1081)
    assert sum(flop.class_distribution().values()) == pytest.approx(1.0)
    # Dead cards (e.g. folded and shown) are removed the same way.
    dead = flop.condition_on_known_cards(dead=C("2c"))
    assert dead.num_live_combos == math.comb(46, 2)


def test_blocked_mask_counts():
    assert blocked_mask(C("As")).sum() == 51
    assert blocked_mask(C("As", "Kd")).sum() == 51 + 51 - 1


@pytest.mark.parametrize("text,classes", [
    ("AA", {"AA"}),
    ("TT+", {"TT", "JJ", "QQ", "KK", "AA"}),
    ("A2s+", {f"A{r}s" for r in "23456789TJQK"}),
    ("KTo+", {"KTo", "KJo", "KQo"}),
    ("AK", {"AKs", "AKo"}),
    ("22-44", {"22", "33", "44"}),
    ("A5s-A3s", {"A5s", "A4s", "A3s"}),
])
def test_parse_range(text, classes):
    assert set(parse_range(text)) == classes


def test_parse_weights_and_errors():
    r = WeightedRange.from_string("AA,KK:0.5")
    assert r.total == pytest.approx(6 + 3)
    assert r.class_mass("AA") == pytest.approx(6 / 9)
    assert WeightedRange.from_string("random").num_live_combos == 1326
    for bad in ("AX", "AAs", "K:Q", "AKx"):
        with pytest.raises(ValueError):
            WeightedRange.from_string(bad)


def test_normalize_entropy_and_sampling():
    u = WeightedRange.uniform()
    assert u.normalize().total == pytest.approx(1.0)
    assert u.entropy_bits() == pytest.approx(math.log2(1326))
    assert u.effective_combos() == pytest.approx(1326)
    aa = WeightedRange.from_string("AA")
    assert aa.effective_combos() == pytest.approx(6)
    rng = np.random.default_rng(0)
    exclude = C("As", "Ks", "Qs", "Js", "4h")
    for a, b in u.sample(rng, 2000, exclude=exclude):
        assert a not in exclude and b not in exclude and a != b
    # Determinism given the seed.
    assert u.sample(np.random.default_rng(5), 10) == \
        u.sample(np.random.default_rng(5), 10)


def test_empty_and_invalid_ranges():
    with pytest.raises(ValueError):
        WeightedRange.empty().probabilities()
    with pytest.raises(ValueError):
        WeightedRange(np.full(NUM_COMBOS, -1.0))
    with pytest.raises(ValueError):
        WeightedRange.from_string("AA").update(
            (~blocked_mask(C("Ah", "Ad", "Ac", "As"))).astype(float))


def test_strength_percentiles_order():
    s = strength_percentiles(C("Ks", "8d", "3c"))
    assert np.isnan(s[combo_index(C("Ks", "Kd"))])     # blocked by board
    kk = s[combo_index(C("Kh", "Kd"))]
    air = s[combo_index(C("7h", "2d"))]
    assert kk > 0.99 and air < 0.3


def test_priors_are_data_driven_and_position_aware():
    p = RangePriors.load()
    utg = p.prior("UTG", "open", 100)
    btn = p.prior("BTN", "open", 100)
    assert utg.num_live_combos < btn.num_live_combos
    assert p.prior("UTG+1", "open").num_live_combos == utg.num_live_combos
    assert p.prior("BTN", "open", 15).num_live_combos != btn.num_live_combos
    assert p.prior("BTN", "nonsense_line").num_live_combos == 1326
    with pytest.raises(ValueError):
        RangePriors({"format": "other"})


def test_classify_preflop_line():
    acts = [A(seat=3, kind="raise"), A(seat=4, kind="call"),
            A(seat=5, kind="raise"), A(seat=3, kind="call")]
    assert classify_preflop_line(acts, 3) == "call_3bet"
    assert classify_preflop_line(acts, 4) == "call_open"
    assert classify_preflop_line(acts, 5) == "3bet"
    assert classify_preflop_line(acts, 1) == "unopened"
    assert classify_preflop_line([A(seat=1, kind="call")], 1) == "limp"


def test_bayesian_update_shifts_mass_to_strong_hands():
    prior = WeightedRange.uniform()
    board = C("Ks", "8d", "3c")
    post = update_range_for_action(prior, "bet", facing_bet=False,
                                   board=board, dead=C("Ah", "Qh"))
    assert post.total == pytest.approx(1.0)
    assert post.weight_of(C("As", "Ah")) == 0.0          # hero blocks
    assert post.weight_of(C("Kh", "Kd")) > post.weight_of(C("7h", "2d"))
    sets = post.class_mass("KK") + post.class_mass("88") + post.class_mass("33")
    assert sets > prior.remove_cards(board).normalize().class_mass("KK") * 3
    check = update_range_for_action(prior, "check", facing_bet=False, board=board)
    assert check.class_mass("KK") < post.class_mass("KK")
    assert RangeBelief.of(post).entropy_bits < RangeBelief.of(
        prior.remove_cards(board)).entropy_bits


def test_likelihood_rows_sum_to_one_for_every_archetype():
    s = np.linspace(0, 1, 101)
    for model in ARCHETYPE_MODELS.values():
        for facing in (False, True):
            probs = model.probabilities(s, facing, 0.75)
            total = sum(probs.values())
            np.testing.assert_allclose(total, 1.0)
            assert all(np.all(v >= -1e-12) for v in probs.values())


def test_strategy_likelihood_with_fallback():
    target = combo_index(C("As", "Ad"))
    lik = StrategyLikelihood(lambda i: {"raise": 1.0} if i == target else None,
                             fallback=BehaviorModel())
    v = lik.likelihood("raise", np.full(NUM_COMBOS, 0.5), facing_bet=True)
    assert v[target] == 1.0 and 0 < v[0] < 1
