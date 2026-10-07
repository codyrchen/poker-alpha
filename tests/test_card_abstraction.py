"""Phase 5: interpretable heads-up Hold'em card abstraction."""

import itertools
from collections import Counter

import numpy as np
import pytest

from poker_alpha.abstraction import (HoldemBucketEncoder, PREFLOP_CLASSES,
                                     RawHoldemEncoder, board_texture,
                                     canonicalize_suits, combos_for_class,
                                     hand_equity, hand_features, preflop_class)
from poker_alpha.abstraction.cards import draw_type, texture_code
from poker_alpha.games import HoldemGame
from poker_alpha.games.holdem import HoldemState
from poker_alpha.poker import card_code
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.serialize import load_checkpoint, save_checkpoint


def C(*cards):
    return [card_code(c) for c in cards]


def permute(cards, perm):
    return [perm[c // 13] * 13 + c % 13 for c in cards]


# -- preflop 169 ---------------------------------------------------------------

def test_all_1326_combos_map_to_exactly_169_classes():
    classes = Counter(preflop_class(c) for c in itertools.combinations(range(52), 2))
    assert sum(classes.values()) == 1326
    assert len(classes) == 169
    assert set(classes) == set(PREFLOP_CLASSES)
    pairs = [c for c in classes if len(c) == 2]
    suited = [c for c in classes if c.endswith("s")]
    offsuit = [c for c in classes if c.endswith("o")]
    assert (len(pairs), len(suited), len(offsuit)) == (13, 78, 78)
    assert all(classes[c] == 6 for c in pairs)
    assert all(classes[c] == 4 for c in suited)
    assert all(classes[c] == 12 for c in offsuit)


def test_combos_for_class_round_trip():
    seen = set()
    for cls in PREFLOP_CLASSES:
        combos = combos_for_class(cls)
        for combo in combos:
            assert preflop_class(combo) == cls
        seen.update(combos)
    assert len(seen) == 1326


def test_preflop_examples_and_suit_symmetry():
    assert preflop_class(["As", "Ah"]) == "AA"
    assert preflop_class(["Ks", "As"]) == "AKs"
    assert preflop_class(["Ad", "Kc"]) == "AKo"
    assert preflop_class(["6h", "7h"]) == "76s"
    for perm in itertools.permutations(range(4)):
        for combo in (C("As", "Ks"), C("7d", "2c"), C("9h", "9c")):
            assert preflop_class(permute(combo, perm)) == preflop_class(combo)


# -- canonicalization ------------------------------------------------------------

def test_canonicalize_suits_identifies_isomorphic_inputs():
    hole, flop = C("As", "Ks"), C("Qs", "7h", "2d")
    base = canonicalize_suits(hole, [flop])
    for perm in itertools.permutations(range(4)):
        assert canonicalize_suits(permute(hole, perm), [permute(flop, perm)]) == base
    # Flop order is irrelevant, but which card came on the turn is not.
    assert canonicalize_suits(hole, [flop[::-1]]) == base
    assert canonicalize_suits(hole, [C("Qs", "7h", "2c")]) == base
    assert canonicalize_suits(hole, [C("Qh", "7s", "2d")]) != base


# -- texture -------------------------------------------------------------------

@pytest.mark.parametrize("board,paired,suited,straight", [
    (("As", "Ks", "Qs"), "unpaired", "monotone", True),
    (("Ah", "Ad", "7c"), "paired", "rainbow", False),
    (("Ah", "Ad", "7d"), "paired", "two_tone", False),
    (("9c", "8d", "2h"), "unpaired", "rainbow", False),
    (("9c", "8d", "7h"), "unpaired", "rainbow", True),
    (("9c", "8c", "2h", "2c"), "paired", "three_flush", False),
    (("Ac", "2d", "3h"), "unpaired", "rainbow", True),  # wheel
    (("Kc", "Kd", "Kh", "2s", "2c"), "full_house", "two_tone", False),
])
def test_board_texture_examples(board, paired, suited, straight):
    t = board_texture(board)
    assert (t.pairedness, t.suitedness, t.straight_possible) == \
        (paired, suited, straight)


def test_texture_invariant_under_suit_permutation():
    board = C("Ts", "9s", "4h", "4d")
    for perm in itertools.permutations(range(4)):
        assert board_texture(permute(board, perm)) == board_texture(board)


# -- hand features -------------------------------------------------------------

@pytest.mark.parametrize("hole,board,draw", [
    (("Ah", "5h"), ("Kh", "9h", "2c"), "flush_draw"),
    (("8c", "7d"), ("6h", "5s", "Kc"), "open_ended"),
    (("8c", "7d"), ("5s", "4h", "Kc"), "gutshot"),
    (("8h", "7h"), ("6h", "5h", "Kc"), "combo_draw"),
    (("Ah", "5h"), ("Kh", "9c", "2c"), "backdoor_flush"),
    (("Ac", "Ad"), ("Kh", "9c", "2s"), "none"),
    (("8h", "7h"), ("6h", "5h", "Kc", "2d", "3c"), "none"),  # river: no draws
])
def test_draw_types(hole, board, draw):
    assert draw_type(C(*hole), C(*board)) == draw


def test_nuts_and_blockers():
    f = hand_features(C("Js", "Ts"), C("As", "Ks", "Qs"))
    assert f.is_nuts and f.hand_category == 8 and f.equity == 1.0
    g = hand_features(C("As", "2d"), C("Ks", "8s", "3s"))
    assert g.flush_blocker and g.nut_flush_blocker and not g.is_nuts
    h = hand_features(C("Qs", "2d"), C("Ks", "8s", "3s"))
    assert h.flush_blocker and not h.nut_flush_blocker
    assert hand_features(C("Ad", "2d"), C("Ks", "8s", "3s")).flush_blocker is False


def test_features_invariant_under_suit_permutation():
    hole, board = C("8h", "7h"), C("6h", "5s", "Kc", "Td")
    base = hand_features(hole, board, pot=10, effective_stack=50)
    for perm in list(itertools.permutations(range(4)))[::5]:
        f = hand_features(permute(hole, perm), permute(board, perm), pot=10,
                          effective_stack=50)
        assert f == base


def test_equity_is_deterministic_and_ordered():
    aa = hand_equity(C("As", "Ad"))
    assert aa == hand_equity(C("Ah", "Ac"))
    kk = hand_equity(C("Ks", "Kd"))
    trash = hand_equity(C("7c", "2d"))
    assert aa > kk > 0.5 > trash
    assert 0.80 < aa < 0.88  # true value ~0.852 vs random


def test_spr_and_position_features():
    f = hand_features(C("As", "Ad"), C("Ks", "8d", "3c"), pot=10,
                      effective_stack=95, position="BTN")
    assert f.spr == pytest.approx(9.5)
    assert f.position == "BTN" and f.effective_stack == 95


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        hand_features(C("As", "As"))
    with pytest.raises(ValueError):
        hand_features(C("As", "Kd"), C("As", "2c", "3d"))
    with pytest.raises(ValueError):
        hand_features(C("As", "Kd"), C("Qs", "2c"))


# -- bucket encoder ------------------------------------------------------------

def _state(holes, board=(), streets=("",)):
    h = [tuple(C(*x)) for x in holes]
    return HoldemState(holes=(h[0], h[1]), board=tuple(C(*board)),
                       streets=tuple(streets), contrib=(0.5, 1.0))


def test_bucket_signature_is_configuration_aware():
    a = HoldemBucketEncoder()
    assert a.signature() == ("HoldemBucketEncoder:v1:equity=10:samples=200:"
                             "spr=1,3,8:texture=v1:history=exact")
    assert HoldemBucketEncoder(equity_buckets=20).signature() != a.signature()
    assert HoldemBucketEncoder(spr_edges=(2,)).signature() != a.signature()
    assert a.signature() != RawHoldemEncoder().signature()


def test_bucket_encoder_merges_suit_isomorphic_states_preflop_and_flop():
    game = HoldemGame(encoder=HoldemBucketEncoder())
    raw = HoldemGame()
    s1 = _state([("As", "Ks"), ("2c", "3d")])
    s2 = _state([("Ah", "Kh"), ("2c", "3d")])
    assert raw.infoset_key(s1) != raw.infoset_key(s2)
    assert game.infoset_key(s1) == game.infoset_key(s2) == "0|AKs|"
    # Postflop the big blind (player 1) acts first.
    f1 = _state([("2c", "3d"), ("As", "Ks")], ("Qs", "7h", "2d"), ("cc", ""))
    f2 = _state([("2s", "3c"), ("Ad", "Kd")], ("Qd", "7h", "2c"), ("cc", ""))
    assert game.infoset_key(f1) == game.infoset_key(f2)
    assert game.infoset_key(f1).startswith("1|AKs/e")


def test_bucket_encoder_preserves_history_and_prior_buckets():
    game = HoldemGame(encoder=HoldemBucketEncoder())
    s = _state([("2c", "3d"), ("As", "Ks")], ("Qs", "7h", "2d", "9c"),
               ("cc", "cc", ""))
    key = game.infoset_key(s)
    player, buckets, history = key.split("|")
    assert history == "cc/cc/"
    assert len(buckets.split("/")) == 3  # preflop, flop, turn
    other = _state([("2c", "3d"), ("As", "Ks")], ("Qs", "7h", "2d", "9c"),
                   ("b100c", "cc", ""))
    assert game.infoset_key(other) != key


def test_bucket_encoder_trains_and_checkpoints(tmp_path):
    enc = HoldemBucketEncoder(equity_samples=50)
    solver = MCCFRSolver(HoldemGame(encoder=enc), seed=1)
    solver.train(3)
    path = save_checkpoint(solver, tmp_path / "b.npz")
    resumed = load_checkpoint(path, HoldemGame(encoder=HoldemBucketEncoder(
        equity_samples=50)))
    assert set(resumed.infosets) == set(solver.infosets)
    with pytest.raises(Exception):
        load_checkpoint(path, HoldemGame(encoder=HoldemBucketEncoder()))
