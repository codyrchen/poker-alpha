"""Tests for the abstracted heads-up NL Hold'em state engine."""

import numpy as np
import pytest

from poker_alpha.games import HoldemGame
from poker_alpha.games.holdem import HoldemState
from poker_alpha.poker import card_code


def make_state(game, holes=("As", "Ah", "Ks", "Kd"), board=(), streets=("",),
               contrib=(0.5, 1.0), folded=-1, all_in=False):
    h = tuple(card_code(c) for c in holes)
    b = tuple(card_code(c) for c in board)
    return HoldemState(holes=((h[0], h[1]), (h[2], h[3])), board=b,
                       streets=tuple(streets), contrib=contrib,
                       folded=folded, all_in=all_in)


@pytest.fixture
def game():
    return HoldemGame()


# -- structure ---------------------------------------------------------------

def test_root_is_chance_and_deal_gives_four_distinct_cards(game):
    root = game.root()
    assert game.is_chance(root)
    s = game.deal(np.random.default_rng(0))
    cards = [c for h in s.holes for c in h]
    assert len(set(cards)) == 4
    assert s.contrib == (0.5, 1.0)


def test_button_acts_first_preflop_bb_first_postflop(game):
    s = make_state(game)
    assert game.current_player(s) == 0
    flop = make_state(game, board=("2c", "7d", "9h"), streets=("cc", ""),
                      contrib=(1.0, 1.0))
    assert game.current_player(flop) == 1


def test_preflop_limp_gives_bb_an_option(game):
    s = make_state(game)
    limp = game.next_state(s, "c")
    assert limp.contrib == (1.0, 1.0)
    assert not game.is_chance(limp)          # BB still to act
    assert game.current_player(limp) == 1
    acts = game.legal_actions(limp)
    assert "f" not in acts                    # nothing to fold to
    assert "c" in acts and "a" in acts


def test_check_check_advances_street(game):
    limp = make_state(game, streets=("c",), contrib=(1.0, 1.0))
    checked = game.next_state(limp, "c")
    assert game.is_chance(checked)            # flop deal pending
    dealt = game.sample_chance(checked, np.random.default_rng(0))
    assert len(dealt.board) == 3
    assert dealt.streets == ("cc", "")


def test_single_postflop_check_does_not_close_street(game):
    flop = make_state(game, board=("2c", "7d", "9h"), streets=("cc", ""),
                      contrib=(1.0, 1.0))
    one_check = game.next_state(flop, "c")
    assert not game.is_chance(one_check)
    assert game.current_player(one_check) == 0


# -- bet sizing --------------------------------------------------------------

def test_pot_bet_sizing_preflop(game):
    s = make_state(game)
    raised = game.next_state(s, "b100")
    # Button calls 0.5 then adds 1.0x the resulting pot (2.0): total 3.0.
    assert raised.contrib[0] == pytest.approx(3.0)
    assert raised.contrib[1] == pytest.approx(1.0)


def test_half_pot_bet_on_flop(game):
    flop = make_state(game, board=("2c", "7d", "9h"), streets=("cc", ""),
                      contrib=(1.0, 1.0))
    bet = game.next_state(flop, "b50")
    # Pot is 2.0; half-pot bet adds 1.0 from the big blind.
    assert bet.contrib == (1.0, 2.0)


def test_all_in_and_call_runs_out_the_board(game):
    s = make_state(game)
    shove = game.next_state(s, "a")
    assert shove.contrib[0] == pytest.approx(100.0)
    assert not game.is_chance(shove)          # BB must respond
    acts = game.legal_actions(shove)
    assert acts == ["f", "c"]                 # cannot raise an all-in
    call = game.next_state(shove, "c")
    assert call.contrib == (100.0, 100.0)
    # Now pure runout: chance until the board has five cards.
    rng = np.random.default_rng(0)
    state = call
    while game.is_chance(state):
        state = game.sample_chance(state, rng)
    assert len(state.board) == 5
    assert game.is_terminal(state)


def test_raise_cap_limits_reraising(game):
    s = make_state(game)
    for action in ("b100", "b100", "b100"):
        assert action in game.legal_actions(s)
        s = game.next_state(s, action)
    acts = game.legal_actions(s)
    assert all(not a.startswith("b") for a in acts)
    assert "a" not in acts                    # cap counts shoves too
    assert "f" in acts and "c" in acts


def test_oversized_bets_are_pruned_to_all_in(game):
    # After heavy raising the pot outgrows the stacks: pot-multiple bets that
    # exceed the remaining stack disappear, leaving only all-in as the top.
    s = make_state(game)
    s = game.next_state(s, "b200")            # pot 1.5 -> button to 5.5
    s = game.next_state(s, "b200")            # reraise
    acts = game.legal_actions(s)
    assert "b200" not in acts
    assert "a" in acts


# -- terminal utilities ------------------------------------------------------

def test_fold_loses_own_contribution(game):
    s = make_state(game)
    raised = game.next_state(s, "b100")       # button to 3.0
    folded = game.next_state(raised, "f")
    assert game.is_terminal(folded)
    assert game.utility(folded) == pytest.approx(1.0)   # BB loses the blind

    folded_btn = game.next_state(make_state(game, streets=("cb100",),
                                            contrib=(1.0, 3.0)), "f")
    assert game.utility(folded_btn) == pytest.approx(-1.0)


def test_showdown_better_hand_wins_pot(game):
    # Board gives player 0 aces over kings; both checked everything.
    s = make_state(game, holes=("As", "Ah", "Ks", "Kd"),
                   board=("2c", "7d", "9h", "3s", "4c"),
                   streets=("cc", "cc", "cc", "cc"),
                   contrib=(1.0, 1.0))
    assert game.is_terminal(s)
    assert game.utility(s) == pytest.approx(1.0)
    # Swap hole cards: utility flips sign.
    s2 = make_state(game, holes=("Ks", "Kd", "As", "Ah"),
                    board=("2c", "7d", "9h", "3s", "4c"),
                    streets=("cc", "cc", "cc", "cc"),
                    contrib=(1.0, 1.0))
    assert game.utility(s2) == pytest.approx(-1.0)


def test_showdown_tie_is_zero(game):
    # Board plays: broadway on board, both hole hands irrelevant.
    s = make_state(game, holes=("2c", "3d", "7s", "8h"),
                   board=("As", "Ks", "Qd", "Jh", "Tc"),
                   streets=("cc", "cc", "cc", "cc"),
                   contrib=(1.0, 1.0))
    assert game.utility(s) == 0.0


# -- information sets --------------------------------------------------------

def test_infoset_hides_opponent_cards(game):
    a = make_state(game, holes=("As", "Ah", "Ks", "Kd"))
    b = make_state(game, holes=("As", "Ah", "2c", "7d"))
    assert game.infoset_key(a) == game.infoset_key(b)


def test_infoset_includes_own_cards_board_and_history(game):
    a = make_state(game, holes=("As", "Ah", "Ks", "Kd"))
    b = make_state(game, holes=("Qs", "Qh", "Ks", "Kd"))
    assert game.infoset_key(a) != game.infoset_key(b)


# -- global invariants over random play --------------------------------------

def test_random_playouts_conserve_chips(game):
    rng = np.random.default_rng(123)
    for _ in range(300):
        s = game.deal(rng)
        steps = 0
        while not game.is_terminal(s):
            if game.is_chance(s):
                s = game.sample_chance(s, rng)
                continue
            acts = game.legal_actions(s)
            s = game.next_state(s, acts[rng.integers(len(acts))])
            steps += 1
            assert steps < 60
        u = game.utility(s)
        assert -s.contrib[0] - 1e-9 <= u <= s.contrib[1] + 1e-9
        assert max(s.contrib) <= game.starting_stack + 1e-9
        if s.folded == -1:
            assert s.contrib[0] == pytest.approx(s.contrib[1])
