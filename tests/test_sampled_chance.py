"""Phase 2: sampled chance as a generic Game capability, used by MCCFR."""

import time

import numpy as np
import pytest

from poker_alpha.games import HoldemGame, KuhnPoker, LeducPoker
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.digest import strategy_digest


class _RecordingHoldem(HoldemGame):
    """Hold'em that records every sampled chance successor."""

    def __init__(self):
        super().__init__()
        self.sampled = []

    def sample_chance(self, state, rng):
        nxt = super().sample_chance(state, rng)
        self.sampled.append(nxt)
        return nxt


def test_default_sample_chance_matches_chance_outcome_support():
    game = LeducPoker()
    rng = np.random.default_rng(0)
    support = {s for _, s in game.chance_outcomes(game.root())}
    for _ in range(50):
        assert game.sample_chance(game.root(), rng) in support


def test_default_sample_chance_consumes_rng_like_legacy_mccfr():
    # Legacy MCCFR did: rng.choice(len(p), p=p/p.sum()) on chance_outcomes.
    game = KuhnPoker()
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    outcomes = game.chance_outcomes(game.root())
    probs = np.array([p for p, _ in outcomes])
    for _ in range(20):
        legacy = outcomes[int(b.choice(len(probs), p=probs / probs.sum()))][1]
        assert game.sample_chance(game.root(), a) == legacy


def test_holdem_mccfr_runs_without_chance_enumeration():
    game = _RecordingHoldem()

    def boom(state):  # chance_outcomes must never be called
        raise AssertionError("chance_outcomes called")

    game.chance_outcomes = boom
    solver = MCCFRSolver(game, seed=0)
    start = time.perf_counter()
    solver.train(10)
    elapsed = time.perf_counter() - start
    assert solver.iterations == 10
    assert len(solver.infosets) > 0
    assert game.sampled
    for s in game.sampled:
        cards = [c for h in s.holes for c in h] + list(s.board)
        assert len(cards) == len(set(cards)), "duplicate card dealt"
        assert all(0 <= c < 52 for c in cards)
        assert len(s.board) in (0, 3, 4, 5)
    print(f"holdem mccfr smoke: 10 iterations, {elapsed:.3f}s, "
          f"{len(solver.infosets)} infosets")


def test_holdem_mccfr_deterministic_given_seed():
    a = MCCFRSolver(HoldemGame(), seed=5)
    a.train(5)
    b = MCCFRSolver(HoldemGame(), seed=5)
    b.train(5)
    assert strategy_digest(a.average_strategy(), None) == \
        strategy_digest(b.average_strategy(), None)
    c = MCCFRSolver(HoldemGame(), seed=6)
    c.train(5)
    assert set(c.infosets) != set(a.infosets)


def test_full_cfr_still_refuses_holdem_enumeration():
    with pytest.raises(NotImplementedError):
        HoldemGame().chance_outcomes(HoldemGame().root())
