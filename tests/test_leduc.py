"""Tests for the Leduc Poker game definition and solver convergence on it."""

import pytest

from poker_alpha.games.leduc import LeducPoker, LeducState
from poker_alpha.solvers import CFRPlusSolver, exploitability, expected_value

# Known value of Leduc poker to player 0 (antes 1, bets 2/4, two raises/round).
LEDUC_VALUE = -0.0856


@pytest.fixture
def game():
    return LeducPoker()


def test_root_deal_has_30_equally_likely_outcomes(game):
    outcomes = game.chance_outcomes(game.root())
    assert len(outcomes) == 30  # 6 * 5 ordered distinct pairs
    assert sum(p for p, _ in outcomes) == pytest.approx(1.0)


def test_public_card_dealt_after_round_one(game):
    s = LeducState((0, 2), None, "cc")
    assert game.is_chance(s)
    outcomes = game.chance_outcomes(s)
    assert len(outcomes) == 4  # six cards minus the two private ones
    dealt = {n.public for _, n in outcomes}
    assert 0 not in dealt and 2 not in dealt
    assert all(n.history == "cc/" for _, n in outcomes)


def test_fold_utilities_are_pot_based(game):
    # P0 raises (2), P1 folds: P1 loses only their ante.
    assert game.utility(LeducState((0, 2), None, "rf")) == 1.0
    # P0 raises, P1 reraises, P0 folds: P0 loses ante + first raise = 3.
    assert game.utility(LeducState((0, 2), None, "rrf")) == -3.0
    # Round 2: cc / r f -> P1 folds with ante only invested.
    assert game.utility(LeducState((0, 2), 4, "cc/rf")) == 1.0


def test_showdown_pair_beats_higher_rank(game):
    # P0 holds J (0), public J (1): pair of jacks beats P1's king.
    assert game.utility(LeducState((0, 2), 1, "cc/cc")) == 1.0
    # Reversed seats: P1 pairs, P0 holds the king.
    assert game.utility(LeducState((2, 0), 1, "cc/cc")) == -1.0


def test_showdown_tie_splits_pot(game):
    # Both players hold jacks of different suits: tie, zero utility.
    assert game.utility(LeducState((0, 1), 4, "cc/cc")) == 0.0
    assert game.utility(LeducState((0, 1), 4, "rc/rc")) == 0.0


def test_showdown_utilities_are_zero_sum_under_seat_swap(game):
    for history in ("cc/cc", "rc/cc", "cc/rc", "rc/rrc"):
        u = game.utility(LeducState((0, 4), 2, history))
        u_swapped = game.utility(LeducState((4, 0), 2, history))
        assert u == -u_swapped


def test_raise_cap_two_per_round(game):
    assert game.legal_actions(LeducState((0, 2), None, "")) == ["c", "r"]
    assert game.legal_actions(LeducState((0, 2), None, "r")) == ["f", "c", "r"]
    assert game.legal_actions(LeducState((0, 2), None, "rr")) == ["f", "c"]
    assert game.legal_actions(LeducState((0, 2), 4, "cc/rr")) == ["f", "c"]


def test_fold_only_legal_facing_a_bet(game):
    assert "f" not in game.legal_actions(LeducState((0, 2), None, ""))
    assert "f" not in game.legal_actions(LeducState((0, 2), 4, "cc/"))
    assert "f" in game.legal_actions(LeducState((0, 2), 4, "cc/r"))


def test_player_zero_acts_first_both_rounds(game):
    assert game.current_player(LeducState((0, 2), None, "")) == 0
    assert game.current_player(LeducState((0, 2), 4, "cc/")) == 0
    assert game.current_player(LeducState((0, 2), 4, "cc/c")) == 1


def test_infoset_hides_opponent_card_and_merges_suits(game):
    # Different opponent cards, same everything else -> same infoset.
    k1 = game.infoset_key(LeducState((0, 2), None, ""))
    k2 = game.infoset_key(LeducState((0, 4), None, ""))
    assert k1 == k2
    # Suit-swapped private card (same rank) -> same infoset.
    k3 = game.infoset_key(LeducState((1, 2), None, ""))
    assert k1 == k3


def test_terminal_detection(game):
    assert game.is_terminal(LeducState((0, 2), None, "rf"))
    assert game.is_terminal(LeducState((0, 2), 4, "cc/cc"))
    assert not game.is_terminal(LeducState((0, 2), None, "cc"))  # chance next
    assert not game.is_terminal(LeducState((0, 2), 4, "cc/c"))


def test_cfr_plus_converges_on_leduc():
    game = LeducPoker()
    solver = CFRPlusSolver(game)
    solver.train(200)
    avg = solver.average_strategy()
    assert expected_value(game, avg) == pytest.approx(LEDUC_VALUE, abs=5e-3)
    assert exploitability(game, avg) < 0.02
