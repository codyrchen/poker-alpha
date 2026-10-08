"""Phase 33/39: exact Hold'em-shaped games and analytic references."""

import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame
from poker_alpha.solvers import CFRSolver, MCCFRSolver
from poker_alpha.solvers.cfr_plus import CFRPlusSolver
from poker_alpha.solvers.evaluation import exploitability

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))


@pytest.fixture(scope="module")
def tiny():
    return ReducedPreflopGame(ranks="AKQ", suits="sh", stack=6.0, equity_samples=200, seed=0)


def test_reduced_preflop_rules(tiny):
    g = tiny
    root = g.chance_outcomes(g.root())[0][1]
    assert g.current_player(root) == 0 and g.legal_actions(root) == ["f", "c", "r", "a"]
    assert g.utility(g.next_state(root, "f")) == -0.5
    s = g.next_state(g.next_state(root, "r"), "f")
    assert g.utility(s) == 1.0
    s = g.next_state(root, "c")
    assert not g.is_terminal(s) and g.legal_actions(s) == ["c", "r", "a"]
    s = g.next_state(s, "c")
    eq = g._equity[root.holes]
    assert g.utility(s) == pytest.approx(eq * 2.0 - 1.0)
    s = g.next_state(g.next_state(root, "a"), "c")
    assert g.utility(s) == pytest.approx(eq * 12.0 - 6.0)
    # equity table is antisymmetric
    for (h0, h1), e in list(g._equity.items())[:50]:
        assert e + g._equity[(h1, h0)] == pytest.approx(1.0)


def test_exact_solvers_and_mccfr_converge(tiny):
    g = tiny
    cfrp = CFRPlusSolver(g)
    cfrp.train(30)
    e30 = exploitability(g, cfrp.average_strategy())
    cfrp.train(270)
    e300 = exploitability(g, cfrp.average_strategy())
    assert e300 < e30 and e300 < 0.01
    cfr = CFRSolver(g)
    cfr.train(300)
    assert exploitability(g, cfr.average_strategy()) < 0.05
    m = MCCFRSolver(g, seed=0)
    m.train(500)
    early = exploitability(g, m.average_strategy())
    m.train(19500)
    late = exploitability(g, m.average_strategy())
    assert late < early and late < 0.05


def test_river_subgame_zero_sum_and_fold_utilities():
    g = RiverSubgame(["Ks", "8d", "4c", "2h", "2s"], ["8s8h", "QsJh"], ["AsKh"], pot=10, stack=20,
                     bets=(0.5,), raise_cap=2)
    d = g.chance_outcomes(g.root())
    assert sum(p for p, _ in d) == pytest.approx(1.0)
    s = d[0][1]
    s1 = g.next_state(g.next_state(s, "b50"), "f")
    assert g.utility(s1) == pytest.approx(5.0)            # OOP wins IP's half of the pot
    s2 = g.next_state(g.next_state(s, "c"), "b50")
    s2 = g.next_state(s2, "f")
    assert g.utility(s2) == pytest.approx(-5.0)


def test_clairvoyance_game_matches_theory():
    from phase39_reference_checks import clairvoyance

    r = clairvoyance(1.0, iters=400)
    assert r["value_bet_freq"] == pytest.approx(1.0, abs=0.02)
    assert r["bluff_share_of_bets"] == pytest.approx(1 / 3, abs=0.03)
    assert r["ip_call_freq"] == pytest.approx(0.5, abs=0.04)
