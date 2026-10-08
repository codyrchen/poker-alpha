"""Phase 28: the optimized Hold'em backend is bit-for-bit equivalent.

Digests below were recorded with the pre-optimization code (commit 5a42e03)
and pin the exact regret / strategy tables after 60 MCCFR iterations (seed
11) for the locked config and three other encoders. Kuhn / Leduc are pinned
separately in test_reproducibility.py.
"""

import hashlib

import numpy as np
import pytest

from poker_alpha.games import HoldemGame
from poker_alpha.games.holdem import _tokens
from poker_alpha.solver_config import PRIMARY_CONFIG, HoldemSolverConfig
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.validation.abstraction_audit import generate_corpus


def exact(s):
    h = hashlib.sha256()
    for k in sorted(s.infosets):
        n = s.infosets[k]
        h.update(k.encode())
        h.update(",".join(n.actions).encode())
        h.update(n.regret_sum.tobytes())
        h.update(n.strategy_sum.tobytes())
    return h.hexdigest()[:16]


PINS = {
    "compact": (lambda: PRIMARY_CONFIG.build_solver(seed=11), 10594, "444d99f40f1bbca1"),
    "bucket": (lambda: HoldemSolverConfig(encoder="bucket").build_solver(seed=11),
               22782, "09041d672f2aa8b1"),
    "transition": (lambda: HoldemSolverConfig(encoder="transition").build_solver(seed=11),
                   22777, "9d1c556a71e62ca7"),
    "raw": (lambda: MCCFRSolver(HoldemGame(), seed=11), 15737, "dfe6e22b236250e5"),
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_training_bit_identical_to_pre_optimization(name):
    make, n, digest = PINS[name]
    s = make()
    s.train(60)
    assert len(s.infosets) == n
    assert exact(s) == digest


def test_fast_sampler_matches_generator_choice():
    s = MCCFRSolver(HoldemGame(), seed=5)
    ref = np.random.default_rng(5)
    gen = np.random.default_rng(1)
    for _ in range(20000):
        k = int(gen.integers(1, 7))
        p = gen.dirichlet(np.ones(k)) if gen.random() < 0.7 else np.full(k, 1.0 / k)
        assert s._sample(p) == int(ref.choice(k, p=p))


def test_memoized_replay_and_legal_match_uncached():
    game = PRIMARY_CONFIG.build_game()
    corpus = generate_corpus(game, 150, 3)
    fresh = PRIMARY_CONFIG.build_game()
    for st in corpus:
        assert game._replay(st) == fresh._replay_uncached(st.streets)
        assert game.legal_actions(st) == fresh._legal_uncached(st)
        assert game.infoset_key(st) == game.infoset_key(st)
    # legal_actions hands out a fresh list each call (callers may mutate it).
    a = game.legal_actions(corpus[0])
    a.append("x")
    assert "x" not in game.legal_actions(corpus[0])


def test_memo_tables_are_bounded(monkeypatch):
    import poker_alpha.games.holdem as H

    monkeypatch.setattr(H, "_MEMO_LIMIT", 8)
    game = PRIMARY_CONFIG.build_game()
    for st in generate_corpus(game, 30, 4):
        game.legal_actions(st)
    assert len(game._replay_memo) <= 8 and len(game._legal_memo) <= 8


def test_tokens_cached_tuple():
    assert _tokens("b33cb75a") == ("b33", "c", "b75", "a")
    assert _tokens("b33cb75a") is _tokens("b33cb75a")
