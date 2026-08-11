"""Tests for the Kuhn Poker game definition."""

import pytest

from poker_alpha.games import KuhnPoker, KuhnState


@pytest.fixture
def game():
    return KuhnPoker()


def _all_decision_states(game):
    """Enumerate every non-terminal decision state (cards dealt)."""
    out = []

    def walk(state):
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for _, s in game.chance_outcomes(state):
                walk(s)
            return
        out.append(state)
        for a in game.legal_actions(state):
            walk(game.next_state(state, a))

    walk(game.root())
    return out


def test_root_is_chance_with_six_equal_deals(game):
    root = game.root()
    assert game.is_chance(root)
    outcomes = game.chance_outcomes(root)
    assert len(outcomes) == 6
    assert sum(p for p, _ in outcomes) == pytest.approx(1.0)
    assert all(p == pytest.approx(1 / 6) for p, _ in outcomes)
    # All deals are two distinct cards.
    assert all(s.cards[0] != s.cards[1] for _, s in outcomes)


def test_terminal_utilities_match_rules(game):
    # King (2) vs Jack (0): player 0 holds the winner.
    assert game.utility(KuhnState((2, 0), "pp")) == 1.0
    assert game.utility(KuhnState((2, 0), "bb")) == 2.0
    assert game.utility(KuhnState((2, 0), "pbb")) == 2.0
    # Jack (0) vs King (2): player 0 holds the loser at showdown.
    assert game.utility(KuhnState((0, 2), "pp")) == -1.0
    assert game.utility(KuhnState((0, 2), "bb")) == -2.0
    # Folds are decided by who folded, not by cards.
    assert game.utility(KuhnState((0, 2), "bp")) == 1.0   # player 1 folded
    assert game.utility(KuhnState((2, 0), "pbp")) == -1.0  # player 0 folded


def test_utility_is_zero_sum(game):
    # Player 1's utility is defined as the negation of player 0's; verify the
    # magnitudes are consistent across all terminal histories for a fixed deal.
    for history in ("pp", "bp", "bb", "pbp", "pbb"):
        u_win = game.utility(KuhnState((2, 0), history))
        u_lose = game.utility(KuhnState((0, 2), history))
        # Swapping the cards flips the result except for fold-decided nodes.
        if history in ("bp", "pbp"):
            assert u_win == u_lose  # fold outcome independent of cards
        else:
            assert u_win == -u_lose


def test_legal_actions_always_two(game):
    for state in _all_decision_states(game):
        assert game.legal_actions(state) == ["p", "b"]


def test_terminal_detection(game):
    for h in ("pp", "bp", "bb", "pbp", "pbb"):
        assert game.is_terminal(KuhnState((2, 0), h))
    for h in ("", "p", "b", "pb"):
        assert not game.is_terminal(KuhnState((2, 0), h))


def test_current_player_alternates(game):
    assert game.current_player(KuhnState((2, 0), "")) == 0
    assert game.current_player(KuhnState((2, 0), "p")) == 1
    assert game.current_player(KuhnState((2, 0), "b")) == 1
    assert game.current_player(KuhnState((2, 0), "pb")) == 0


def test_information_set_hides_opponent_card(game):
    # At history "b" player 1 acts. Vary player 0's (opponent's) card while
    # holding player 1's card and the history fixed => same information set.
    k1 = game.infoset_key(KuhnState((2, 0), "b"))  # player 1 holds 0, opp holds 2
    k2 = game.infoset_key(KuhnState((1, 0), "b"))  # player 1 holds 0, opp holds 1
    assert k1 == k2 == "0b"


def test_information_set_key_uses_acting_player_card(game):
    # At history "" player 0 acts and sees their own card.
    assert game.infoset_key(KuhnState((2, 1), "")) == "2"
    # At history "b" player 1 acts and sees their own card (index 1).
    assert game.infoset_key(KuhnState((2, 1), "b")) == "1b"
    assert game.infoset_key(KuhnState((0, 1), "pb")) == "0pb"  # player 0 acts
