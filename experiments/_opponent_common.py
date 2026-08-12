"""Shared plumbing for the opponent-modeling headline experiments.

Not part of the ``poker_alpha`` package API: this is experiment-only glue
(solving the reference equilibrium, building the archetype roster, and a
seat-averaged exact-EV helper) reused across ``adaptation_vs_archetypes.py``,
``opponent_identification.py``, ``overfitting_vs_sample_size.py``, and
``regime_change.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from poker_alpha.games.leduc import LeducPoker
from poker_alpha.opponent import ARCHETYPES, leduc_is_weak, tilt_strategy
from poker_alpha.solvers import CFRPlusSolver
from poker_alpha.solvers.evaluation import profile_value

Strategy = Dict[str, Dict[str, float]]


def solve_leduc_equilibrium(iterations: int = 300) -> Tuple[LeducPoker, Strategy]:
    """Train CFR+ to a near-equilibrium Leduc strategy (same budget as tests)."""
    game = LeducPoker()
    solver = CFRPlusSolver(game)
    solver.train(iterations)
    return game, solver.average_strategy()


def build_archetypes(equilibrium: Strategy) -> Dict[str, Strategy]:
    """Tilt the equilibrium into every named archetype (balanced = identity)."""
    return {name: tilt_strategy(equilibrium, cfg, leduc_is_weak)
            for name, cfg in ARCHETYPES.items()}


def seat_avg_ev(game: LeducPoker, hero: Strategy, opponent: Strategy) -> float:
    """Exact hero EV (chips/hand) averaged over both seats.

    Both strategies are static, so this is a full tree traversal with no
    sampling noise — the ceiling/floor references the Monte Carlo match
    results are compared against.
    """
    return 0.5 * (profile_value(game, hero, opponent)
                  - profile_value(game, opponent, hero))
