"""Phase 35B: legal-NLHE sizing option (HoldemSolverConfig v2)."""

import numpy as np
import pytest

from poker_alpha.games.holdem import _tokens
from poker_alpha.solver_config import LEGAL_SIZING_CONFIG, PRIMARY_CONFIG
from poker_alpha.solvers.holdem_analysis import action_label, spot_state
from poker_alpha.validation.abstraction_audit import generate_corpus


def test_v1_signature_and_game_unchanged():
    assert PRIMARY_CONFIG.signature() == "HoldemSolverConfig:v1:2e23c098860f7894c871"
    assert ":preflop=" not in PRIMARY_CONFIG.build_game().signature()
    assert LEGAL_SIZING_CONFIG.signature().startswith("HoldemSolverConfig:v2:")
    assert LEGAL_SIZING_CONFIG.signature() != PRIMARY_CONFIG.signature()


def test_preflop_sizes_are_multiples_of_the_current_bet():
    g = LEGAL_SIZING_CONFIG.build_game()
    s = spot_state(g, "BTN", ("As", "Ah"), (), ("",), villain_hole=("7c", "2d"))
    to = {a: g.next_state(s, a).contrib[0] for a in g.legal_actions(s)}
    assert to == pytest.approx({"f": 0.5, "c": 1.0, "x200": 2.0, "x250": 2.5, "x350": 3.5, "a": 100.0})
    s = spot_state(g, "BB", ("As", "Ah"), (), ("x250",), villain_hole=("7c", "2d"))
    to = {a: g.next_state(s, a).contrib[1] for a in g.legal_actions(s)}
    assert to["x200"] == pytest.approx(5.0) and to["x350"] == pytest.approx(8.75)
    assert action_label(g, s, "x250") == "raise_to_2.5x"
    assert _tokens("x250x350c") == ("x250", "x350", "c")


def _increments(game, st):
    """(offered raise increment, NLHE minimum) for every non-all-in raise."""
    street_paid, total, me, _ = game._replay(st)
    owe = street_paid[1 - me] - street_paid[me]
    for tok in game.legal_actions(st):
        if tok in ("f", "c", "a"):
            continue
        add = game.next_state(st, tok).contrib[me] - st.contrib[me]
        yield tok, add - owe, game._min_increment(st.streets)


def test_no_offered_bet_or_raise_below_nlhe_minimum():
    g = LEGAL_SIZING_CONFIG.build_game()
    n = 0
    for st in generate_corpus(g, 400, 5):
        for tok, inc, mn in _increments(g, st):
            n += 1
            assert inc >= mn - 1e-9, (st.streets, tok, inc, mn)
            assert mn >= 1.0 - 1e-12
    assert n > 1000


def test_v1_still_offers_sub_minimum_sizes():
    """The v1 game is unchanged: its 33% sizes can be below the minimum."""
    g = PRIMARY_CONFIG.build_game()
    below = sum(1 for st in generate_corpus(g, 200, 5) for _, inc, mn in _increments(g, st)
                if inc < mn - 1e-9)
    assert below > 0


def test_v2_chip_conservation_and_terminal_utilities():
    g = LEGAL_SIZING_CONFIG.build_game()
    rng = np.random.default_rng(1)
    for _ in range(300):
        s = g.deal(rng)
        while not g.is_terminal(s):
            if g.is_chance(s):
                s = g.sample_chance(s, rng)
                continue
            _, total, _, _ = g._replay(s)
            assert s.contrib == pytest.approx(total)
            assert max(total) <= g.starting_stack + 1e-9
            legal = g.legal_actions(s)
            s = g.next_state(s, legal[rng.integers(len(legal))])
        assert abs(g.utility(s)) <= g.starting_stack + 1e-9


def test_v2_trains_and_keys_carry_legal_menu():
    s = LEGAL_SIZING_CONFIG.build_solver(seed=0)
    s.train(20)
    assert any("x250" in k for k in s.infosets)
    # no preflop key offers the postflop pot-fraction tokens
    assert not any(k.startswith("0|") and ".b33" in k for k in s.infosets)
