"""Tabular-tree native adapter + counterfactual action values (final-trust
Phase 3/6 infrastructure)."""

from __future__ import annotations

import numpy as np
import pytest

from poker_alpha.games.reduced_holdem import ReducedPreflopGame, RiverSubgame
from poker_alpha.solvers.action_values import (action_margin, cf_action_values,
                                               policy_regret)
from poker_alpha.solvers.cfr_plus import CFRPlusSolver
from poker_alpha.solvers.evaluation import expected_value, exploitability

pytest.importorskip("poker_alpha_native")

from poker_alpha.native.tabular import NativeTabularMCCFR, build_tabular_tree  # noqa: E402


@pytest.fixture(scope="module")
def tiny_game():
    return ReducedPreflopGame(ranks="AKQ", suits="sh", stack=10.0,
                              equity_samples=300, seed=0)


def test_tree_matches_python_game(tiny_game):
    """Tree utilities and keys are literally the Python game's."""
    t = build_tabular_tree(tiny_game)
    assert t.player[0] == 0                      # button acts first
    # Every terminal utility equals a direct Python replay for 50 random deals.
    rng = np.random.default_rng(0)
    outcomes = tiny_game.chance_outcomes(tiny_game.root())
    terminals = [n for n in range(len(t.player)) if t.term_idx[n] >= 0]
    n0, n1 = len(t.hands0), len(t.hands1)
    util = t.util.reshape(len(terminals), n0, n1)
    h0i = {h: i for i, h in enumerate(t.hands0)}
    h1i = {h: i for i, h in enumerate(t.hands1)}
    for _ in range(50):
        _, state = outcomes[int(rng.integers(len(outcomes)))]
        tn = terminals[int(rng.integers(len(terminals)))]
        s = state
        for a in t.node_paths[tn]:
            s = tiny_game.next_state(s, a)
        expect = tiny_game.utility(s)
        got = util[t.term_idx[tn], h0i[state.holes[0]], h1i[state.holes[1]]]
        assert got == expect


def test_native_tabular_converges_on_exact_game(tiny_game):
    ref = CFRPlusSolver(tiny_game)
    ref.train(600)
    ref_expl = exploitability(tiny_game, ref.average_strategy())
    assert ref_expl < 0.01

    solver = NativeTabularMCCFR(tiny_game, seed=0)
    solver.train(2_000)
    early = exploitability(tiny_game, solver.average_strategy())
    solver.train(48_000)
    late = exploitability(tiny_game, solver.average_strategy())
    assert late < early                 # converging
    assert late < 0.05                  # near the exact solution


def test_native_tabular_river_subgame():
    g = RiverSubgame(["Ks", "7d", "2c", "Qh", "4s"],
                     ["AsAd", "AhAc", "KdKh", "2h7h", "3c3d"],
                     ["QdQs", "JcJd", "8h8s", "5c5d"],
                     pot=10.0, stack=20.0, bets=(0.75,), raise_cap=1)
    solver = NativeTabularMCCFR(g, seed=1)
    solver.train(30_000)
    expl = exploitability(g, solver.average_strategy())
    assert expl < 0.3                   # pot is 10; near-equilibrium

    sv = solver.average_strategy_with_visits()
    assert all(v >= 0 for _, v in sv.values())
    assert sum(v for _, v in sv.values()) > 0


def test_cf_action_values_identities(tiny_game):
    ref = CFRPlusSolver(tiny_game)
    ref.train(600)
    strat = ref.average_strategy()
    q = cf_action_values(tiny_game, strat)
    assert len(q) > 10
    value = expected_value(tiny_game, strat)
    # Reach-weighted sum of per-player node values reproduces consistency:
    # at an equilibrium-ish profile, per-infoset regret must be ~0.
    regrets = [policy_regret(info["q"], strat.get(k)) for k, info in q.items()]
    assert np.median(regrets) < 0.02
    # Sanity: q values are bounded by the stack.
    for info in q.values():
        for v in info["q"].values():
            assert abs(v) <= tiny_game.stack + 1e-9
        assert action_margin(info["q"]) >= 0
    assert np.isfinite(value)


def test_policy_regret_uniform_fallback():
    q = {"f": 0.0, "c": 1.0, "r": -0.5}
    assert policy_regret(q, None) == pytest.approx(1.0 - (0.5 / 3))
    assert policy_regret(q, {"c": 1.0}) == pytest.approx(0.0)
    assert action_margin(q) == pytest.approx(1.0)  # best 1.0 - second 0.0
