"""Phase 26: cheap card features, betting-history abstraction, v2 encoders."""

import itertools

import pytest

from poker_alpha.abstraction import HoldemBucketEncoder, RawHoldemEncoder
from poker_alpha.abstraction.betting_history import betting_context, size_class
from poker_alpha.abstraction.features import (MADE_STRENGTH, card_features,
                                              made_hand_class, transition_label)
from poker_alpha.abstraction.holdem_v2 import (CompactHoldemEncoder,
                                               TransitionHoldemEncoder)
from poker_alpha.games import HoldemGame
from poker_alpha.games.holdem import HoldemState
from poker_alpha.poker import card_code
from poker_alpha.validation.abstraction_audit import (audit_perfect_recall,
                                                      generate_corpus,
                                                      generate_line_corpus,
                                                      measure_compression,
                                                      measure_quality,
                                                      state_facts)

BETS = {"b33": 0.33, "b75": 0.75, "b150": 1.5}


def C(s):
    return [card_code(x) for x in s.split()]


@pytest.mark.parametrize("hole,board,made", [
    ("As Kd", "Ah 7c 2d", "top_pair_good"),
    ("As 9d", "Ah 7c 2d", "top_pair_weak"),
    ("Qs Qd", "Jh 7c 2d", "overpair"),
    ("5s 5d", "Jh 7c 6d", "underpair"),
    ("9s 9d", "Jh 7c 6d", "pocket_middle"),
    ("7s 7d", "Ah 7c 2d", "set"),
    ("As 7d", "7h 7c 2d", "trips"),
    ("As 7d", "Ah 7c 2d", "two_pair"),
    ("Ks Qd", "Jh 7c 2d", "overcards"),
    ("Ks 3d", "Jh 7c 2d", "air_high"),
    ("8s 3d", "Jh 7c 2d", "air"),
    ("Ad 3c", "Jh 7c 2d", "ace_high"),
    ("Jd 7d", "Jh 7c 2d", "two_pair"),
    ("7d 3s", "Jh 7c 2d", "middle_pair"),
    ("2s 3s", "Jh 7c 2d", "bottom_pair"),
    ("2c 3d", "Ah Ad As Ac Kd", "board_plays"),
    ("9h 8h", "Th Jh Qc 2c 3d", "straight"),
    ("Ah 2h", "Th Jh 4h 5c 3d", "flush"),
])
def test_made_hand_classes(hole, board, made):
    assert made_hand_class(C(hole), C(board)) == made
    assert made in MADE_STRENGTH


def test_features_are_suit_invariant_and_cached():
    base = card_features(C("As Kd"), C("Ah 7c 2d"))
    for perm in itertools.permutations(range(4)):
        h = [perm[c // 13] * 13 + c % 13 for c in C("As Kd")]
        b = [perm[c // 13] * 13 + c % 13 for c in C("Ah 7c 2d")]
        f = card_features(h, b)
        assert (f.made, f.strength, f.draw, f.nut, f.texture) == \
            (base.made, base.strength, base.draw, base.nut, base.texture)
    assert card_features(C("Kd As"), C("2d Ah 7c")) is base   # lru_cache hit


def test_draws_nuts_blockers_potential():
    combo = card_features(C("8h 9h"), C("Th Jh 2c"))
    assert combo.draw == 3 and combo.potential == 2 and combo.blocker == 1
    river = card_features(C("8h 9h"), C("Th Jh 2c 3s 4d"))
    assert river.draw == 0 and river.potential == 0
    nuts = card_features(C("As Ad"), C("Ah 7c 2d"))        # top set, no draws possible
    assert nuts.made == "set" and nuts.nut == 2
    assert card_features(C("7s 7d"), C("Ah 7c 2d")).nut == 1
    assert card_features(C("Kh Qh"), C("Ah 7h 2h")).blocker == 2  # holds the K: nut-flush card missing is K
    assert card_features(C("As Ks")).street == 0


def test_transition_labels():
    flop = card_features(C("8h 9h"), C("Th Jh 2c"))
    made = card_features(C("8h 9h"), C("Th Jh 2c Qs"))      # straight completes
    assert transition_label(flop, made).startswith("Ic")
    miss = card_features(C("8h 9h"), C("Th Jh 2c 3s 4d"))
    turn = card_features(C("8h 9h"), C("Th Jh 2c 3s"))
    assert transition_label(turn, miss)[1] == "m"
    paired = card_features(C("As Kd"), C("Ah 7c 2d 7s"))
    assert transition_label(card_features(C("As Kd"), C("Ah 7c 2d")), paired)[-1] == "T"


def test_size_classes():
    assert size_class(0.33, False) == "small"
    assert size_class(0.75, False) == "medium"
    assert size_class(1.0, False) == "large"
    assert size_class(1.5, False) == "over"
    assert size_class(0.2, True) == "allin"


def state(streets, board=""):
    h = C("As Kd") + C("7c 2h")
    return HoldemState(holes=((h[0], h[1]), (h[2], h[3])), board=tuple(C(board)) if board else (),
                       streets=tuple(streets), contrib=(0.0, 0.0))


def test_betting_context():
    g = HoldemGame(bet_fractions=BETS)
    pre = betting_context(g, state(("",)))
    assert (pre.position, pre.initiative, pre.raises, pre.facing) == (0, "none", 0, "none")
    assert pre.legal.startswith("f.c")
    facing = betting_context(g, state(("b75",)))
    assert facing.position == 1 and facing.facing == "medium" and facing.raises == 1
    flop = betting_context(g, state(("b75c", "cb150"), "Ah 7c 2d"))
    assert flop.position == 1 and flop.initiative == "opp"        # BTN raised preflop
    assert flop.facing == "over" and flop.own_prior == "check"
    btn = betting_context(g, state(("b75c", "c"), "Ah 7c 2d"))
    assert btn.position == 0 and btn.initiative == "own" and btn.facing == "none"


def test_signatures_are_distinct_and_flag_recall():
    sigs = {TransitionHoldemEncoder().signature(), TransitionHoldemEncoder("abstract").signature(),
            CompactHoldemEncoder().signature(), CompactHoldemEncoder("exact").signature(),
            CompactHoldemEncoder(texture=False).signature()}
    assert len(sigs) == 5
    assert "recall=imperfect" in CompactHoldemEncoder().signature()
    assert TransitionHoldemEncoder().perfect_recall_by_design
    assert not TransitionHoldemEncoder("abstract").perfect_recall_by_design
    assert not CompactHoldemEncoder().perfect_recall_by_design
    with pytest.raises(ValueError):
        CompactHoldemEncoder("weird")


@pytest.fixture(scope="module")
def corpus():
    g = HoldemGame(bet_fractions=BETS)
    return g, generate_corpus(g, 200, seed=5)


ENCODERS = {"transition": TransitionHoldemEncoder(), "transition_abs": TransitionHoldemEncoder("abstract"),
            "compact": CompactHoldemEncoder(), "compact_exact": CompactHoldemEncoder("exact")}


def test_v2_encoders_are_functions_of_raw_infostate_with_single_legal_menu(corpus):
    g, c = corpus
    rep = measure_compression(g, c, ENCODERS)
    facts = state_facts(g, c, equity_samples=20)
    for name, enc in ENCODERS.items():
        assert rep.invariant_violations[name] == 0
        assert rep.keys[name] <= rep.raw_keys
        q = measure_quality(g, c, facts, enc)
        assert q["share_of_states_in_keys_mixing"]["legal"] == 0.0, name
    assert rep.keys["compact"] < rep.keys["transition"]


def test_recall_status(corpus):
    g, c = corpus
    assert audit_perfect_recall(g, c, TransitionHoldemEncoder(), "t").violations == 0
    compact = audit_perfect_recall(g, c, CompactHoldemEncoder(), "c")
    assert compact.violations > 0          # imperfect recall, quantified


def test_compact_compresses_fixed_flop_line():
    g = HoldemGame(bet_fractions=BETS)
    lc = generate_line_corpus(g, ("b75c", ""), 600, seed=1)
    rep = measure_compression(g, lc, {"compact": CompactHoldemEncoder(),
                                      "bucket": HoldemBucketEncoder(equity_samples=30)})
    assert rep.keys["compact"] * 4 < rep.keys["bucket"]
