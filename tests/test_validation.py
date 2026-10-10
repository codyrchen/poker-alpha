"""Phase 21: cross-cutting validation (property-style and known answers)."""

import numpy as np
import pytest

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.holdem import (Action, IllegalActionError, Street,
                                TableConfig, apply_action, build_pots,
                                cards_needed, check_invariants,
                                deal_board_random, known_cards, legal_actions,
                                settle, start_hand)
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.poker import card_code
from poker_alpha.poker.evaluator import evaluate_best_codes
from poker_alpha.poker.ranges import (CARD_MASK, COMBOS, NUM_COMBOS,
                                      WeightedRange, blocked_mask)


# -- rules engine properties ----------------------------------------------------

def _random_action(rng, la):
    opts = []
    if la.can_fold:
        opts.append(Action.fold())
    opts.append(Action.check() if la.can_check else Action.call())
    if la.can_raise:
        to = int(rng.integers(la.min_raise_to, la.max_raise_to + 1))
        opts.append(Action.bet(to) if la.is_bet else Action.raise_to(to))
        opts.append(Action.all_in())
    return opts[int(rng.integers(len(opts)))]


@pytest.mark.parametrize("seed", range(4))
def test_engine_properties_on_random_hands(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(120):
        n = int(rng.integers(2, 10))
        stacks = [int(x) for x in rng.integers(3, 500, size=n)]
        deck = rng.permutation(52)
        holes = [(int(deck[2 * i]), int(deck[2 * i + 1])) for i in range(n)]
        s = start_hand(stacks, int(rng.integers(n)),
                       TableConfig(1, 2, int(rng.integers(0, 3))), hole_cards=holes)
        total = sum(stacks)
        while not s.is_complete:
            check_invariants(s, total)
            # pot equals contributions before awards
            assert s.pot == sum(x.committed_total for x in s.seats)
            # no duplicate cards
            cards = known_cards(s)
            assert len(cards) == len(set(cards))
            if s.actor is not None:
                assert s.seats[s.actor].can_act      # only an active player acts
                s = apply_action(s, _random_action(rng, legal_actions(s)))
            elif cards_needed(s):
                s = deal_board_random(s, rng)
            else:
                with pytest.raises(IllegalActionError):
                    legal_actions(s)                 # nobody may act now
                s = settle(s)
        # terminal: valid award, conservation, side-pot eligibility
        assert sum(x.stack for x in s.seats) == total
        contrib = [x.committed_total for x in s.seats]
        assert sum(w for _, w in s.awards) == sum(contrib)
        live = [x.seat for x in s.seats if not x.folded]
        pots = build_pots(contrib, [x.folded for x in s.seats])
        won = dict(s.awards)
        if len(live) > 1:
            values = {i: evaluate_best_codes(list(s.seats[i].hole_cards) + list(s.board))
                      for i in live}
            for seat, amount in won.items():
                assert seat in live
                eligible_pots = [p for p in pots if seat in p.eligible]
                assert eligible_pots, "winner not eligible for any pot"
                assert any(values[seat] == max(values[e] for e in p.eligible)
                           for p in eligible_pots)
                assert amount <= sum(p.amount for p in eligible_pots)
        else:
            assert list(won) == live


# -- range properties -------------------------------------------------------------

def test_range_properties_random():
    rng = np.random.default_rng(5)
    for _ in range(200):
        w = rng.random(NUM_COMBOS) * (rng.random(NUM_COMBOS) < 0.3)
        r = WeightedRange(w)
        if r.is_empty():
            continue
        assert r.probabilities().sum() == pytest.approx(1.0)
        dead = [int(c) for c in rng.choice(52, size=int(rng.integers(1, 8)), replace=False)]
        b = r.remove_cards(dead)
        mask = blocked_mask(dead)
        assert np.all(b.weights[mask] == 0)
        assert np.array_equal(b.weights[~mask], r.weights[~mask])
        if not b.is_empty():
            assert b.probabilities().sum() == pytest.approx(1.0)
            for a, c in b.sample(rng, 50, exclude=dead):
                assert a != c and a not in dead and c not in dead
    # exact combinatorics of card removal
    for k in range(0, 8):
        dead = list(range(k))
        assert int((~blocked_mask(dead)).sum()) == (52 - k) * (51 - k) // 2


# -- decision known answers ---------------------------------------------------

def hu(hero, board, *, opp_bet=0.0, pot, opp_stack=100.0, opp_all_in=False,
       hero_stack=100.0, hero_bet=0.0):
    return ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 1, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": hero, "board": board, "pot": pot,
        "actor": 1, "seats": [{"stack": opp_stack, "bet": opp_bet,
                               "all_in": opp_all_in},
                              {"stack": hero_stack, "bet": hero_bet}]})


def test_pot_odds_boundary():
    # Hero (KK) beats exactly one third of the villain's all-in river range.
    board = "2c 7d 9s Th 3h"
    C = card_code
    villain = {0: WeightedRange.from_combos([((C("Ac"), C("Ad")), 1),
                                             ((C("Ah"), C("As")), 1),
                                             ((C("4c"), C("4d")), 1)])}
    cfg = DecisionConfig(equity_simulations=6000, seed=3)
    # Call 20 into 120 -> needs 14%: call. Call 100 into 110 -> needs 48%: fold.
    good = recommend_action(hu("Kc Kd", board, opp_bet=20, pot=120, opp_stack=0,
                               opp_all_in=True), opponent_ranges=villain, config=cfg)
    bad = recommend_action(hu("Kc Kd", board, opp_bet=100, pot=110, opp_stack=0,
                              opp_all_in=True), opponent_ranges=villain, config=cfg)
    assert good.hero_equity == pytest.approx(1 / 3, abs=0.03)
    assert good.pot_odds == pytest.approx(20 / 140)
    assert good.recommended == "call" and good.candidate("call").ev_bb > 0
    assert bad.pot_odds == pytest.approx(100 / 210)
    assert bad.recommended == "fold" and bad.candidate("call").ev_bb < 0


def test_forced_all_in_call_and_free_check_known_answers():
    cfg = DecisionConfig(equity_simulations=500, seed=0)
    nuts = recommend_action(hu("As Ah", "Ad Ac 2s 7h 9d", opp_bet=50, pot=60,
                               opp_stack=0, opp_all_in=True), config=cfg)
    assert nuts.hero_equity == 1.0 and nuts.recommended == "call"
    free = recommend_action(hu("7c 2d", "Ah Kh Qd Js 4c", pot=10), config=cfg)
    assert free.recommended == "check" and "fold" not in [c.label for c in free.candidates]


# -- observer (synthetic only) -----------------------------------------------------

@pytest.mark.vision
def test_observer_synthetic_metrics_and_false_events():
    pytest.importorskip("PIL")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from observer_validation import false_events  # noqa: E402

    from poker_alpha.observer.evaluation import evaluate
    from poker_alpha.observer.pokernow import (PokerNowStyleAdapter,
                                               default_layout)
    from poker_alpha.observer.synthetic import random_table, render_table

    rng = np.random.default_rng(11)
    cal = default_layout(6, 0)
    ad = PokerNowStyleAdapter(cal)
    tables = [random_table(rng, 6) for _ in range(10)]
    rep = evaluate(ad, [(render_table(t, cal, seed=i), t) for i, t in enumerate(tables)])
    assert rep.card_accuracy == 1.0 and rep.numeric_accuracy >= 0.97
    # Watching an unchanging table must not invent events.
    fe = list(false_events(ad, cal, tables[0], 8, 3.0, (1280, 800), 0))
    assert sum(fe) == 0
