"""Phase II: signaling game, cheap-talk oracle, generic complexity inspector."""

import pytest

from poker_alpha.games.inspect import inspect_game
from poker_alpha.games.kuhn import KuhnPoker
from poker_alpha.games.leduc import LeducPoker
from poker_alpha.games.signaling import SignalingGame
from poker_alpha.research.cfr_variants import DiscountedCFRSolver
from poker_alpha.solvers.cfr_plus import CFRPlusSolver
from poker_alpha.solvers.evaluation import expected_value, exploitability

PAYOFF = ((2.0, -1.0, -1.0), (-1.0, 2.0, -1.0), (-1.0, -1.0, 2.0))


def test_cheap_talk_value_equals_no_information_value():
    g = SignalingGame((0.5, 0.3, 0.2), PAYOFF, None, 3)
    s = CFRPlusSolver(g)
    s.train(3000)
    strat = s.average_strategy()
    assert exploitability(g, strat) < 1e-4
    assert -expected_value(g, strat) == pytest.approx(g.no_information_value_receiver(), abs=1e-3)


def test_costly_signals_can_be_informative():
    g = SignalingGame((0.1, 0.9), ((1.0, -1.0), (-1.0, 1.0)), ((0.0, 0.6), (0.6, 0.0)), 2)
    s = DiscountedCFRSolver.preset(g, "dcfr")
    s.train(3000)
    strat = s.average_strategy()
    assert exploitability(g, strat) < 1e-3
    assert -expected_value(g, strat) > g.no_information_value_receiver() + 0.03


def test_free_message_lets_sender_pool():
    cost = ((0.0, 1.5, 3.0), (0.0, 0.0, 1.5), (0.0, 0.0, 0.0))
    g = SignalingGame((0.5, 0.3, 0.2), PAYOFF, cost, 3)
    s = CFRPlusSolver(g)
    s.train(3000)
    assert -expected_value(g, s.average_strategy()) == pytest.approx(g.no_information_value_receiver(), abs=1e-3)


def test_invalid_signaling_configs():
    with pytest.raises(ValueError):
        SignalingGame((0.5, 0.6))
    with pytest.raises(ValueError):
        SignalingGame((0.5, 0.5), PAYOFF)
    with pytest.raises(ValueError):
        SignalingGame((0.5, 0.5), ((1, 0), (0, 1)), ((0.0,), (0.0,)), 2)


def test_inspector_counts_known_games():
    k = inspect_game(KuhnPoker())
    assert k["ok"] and k["infosets"] == 12 and k["terminal_nodes"] == 30 and k["chance_nodes"] == 1
    lg = inspect_game(LeducPoker())
    assert lg["ok"] and lg["infosets"] == 288 and lg["infosets_by_player"] == {"0": 144, "1": 144}
    sg = inspect_game(SignalingGame())
    assert sg["ok"] and sg["infosets"] == 4 and sg["max_depth"] == 3


def test_inspector_flags_broken_game():
    class Broken(KuhnPoker):
        def infoset_key(self, s):
            return "same"
    rep = inspect_game(Broken())
    assert not rep["ok"] and any("shared by both players" in p or "inconsistent" in p for p in rep["problems"])
