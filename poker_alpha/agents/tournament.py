"""Reproducible agent tournaments on generic extensive-form games.

Round-robin with seat swapping (every pairing plays both seat orders on
the SAME chance seeds, which pairs out most deal luck), fixed seeds, and
normal-approximation confidence intervals. Utilities are the game's own
units (Kuhn: chips per hand).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..games.base import Game
from .base import Agent


def play_episode(game: Game, agents: Sequence[Agent],
                 rng: random.Random,
                 chance_rng: random.Random = None) -> float:
    """One hand/episode; returns utility to player 0. ``chance_rng``
    (when given) drives only the chance nodes, so two episodes sharing it
    see the same deals regardless of the agents' choices at matching
    chance depths."""
    chance_rng = chance_rng or rng
    state = game.root()
    while not game.is_terminal(state):
        if game.is_chance(state):
            outcomes = game.chance_outcomes(state)
            r = chance_rng.random()
            acc = 0.0
            for p, nxt in outcomes:
                acc += p
                if r <= acc:
                    state = nxt
                    break
            else:
                state = outcomes[-1][1]
            continue
        player = game.current_player(state)
        legal = game.legal_actions(state)
        a = agents[player].choose_action(game.infoset_key(state), legal, rng)
        if a not in legal:
            raise ValueError(f"agent {player} chose illegal action {a!r}")
        state = game.next_state(state, a)
    return game.utility(state)


@dataclass(frozen=True)
class MatchResult:
    agent_a: str
    agent_b: str
    hands: int
    mean_utility_a: float      # per hand, seat-swapped
    se: float

    @property
    def ci95(self) -> Tuple[float, float]:
        return (self.mean_utility_a - 1.96 * self.se,
                self.mean_utility_a + 1.96 * self.se)


def play_match(game: Game, a: Agent, b: Agent, hands: int,
               seed: int = 0) -> MatchResult:
    """``hands`` seat-swapped pairs: each chance seed is played once with
    (a,b) and once with (b,a); the paired difference removes deal luck."""
    utils: List[float] = []
    for h in range(hands):
        a.reset(); b.reset()
        u_ab = play_episode(game, (a, b),
                            random.Random(hash((seed, h, "ab")) & 0x7FFFFFFF),
                            random.Random(hash((seed, h, "deal")) & 0x7FFFFFFF))
        u_ba = play_episode(game, (b, a),
                            random.Random(hash((seed, h, "ba")) & 0x7FFFFFFF),
                            random.Random(hash((seed, h, "deal")) & 0x7FFFFFFF))
        utils.append(0.5 * (u_ab - u_ba))   # utility to `a`, seat-averaged
    n = len(utils)
    mean = sum(utils) / n
    var = sum((u - mean) ** 2 for u in utils) / max(n - 1, 1)
    return MatchResult(type(a).__name__, type(b).__name__, n, mean,
                       math.sqrt(var / n))


def round_robin(game: Game, agents: Dict[str, Agent], hands: int = 2000,
                seed: int = 0) -> dict:
    names = list(agents)
    matrix: Dict[str, Dict[str, dict]] = {n: {} for n in names}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            r = play_match(game, agents[na], agents[nb], hands,
                           seed=seed + i * 1000)
            lo, hi = r.ci95
            matrix[na][nb] = {"mean": round(r.mean_utility_a, 4),
                              "se": round(r.se, 4),
                              "ci95": [round(lo, 4), round(hi, 4)],
                              "hands": r.hands}
            matrix[nb][na] = {"mean": round(-r.mean_utility_a, 4),
                              "se": round(r.se, 4),
                              "ci95": [round(-hi, 4), round(-lo, 4)],
                              "hands": r.hands}
    return {
        "format": "pokeralpha.tournament/v1",
        "game": type(game).__name__, "hands_per_pairing": hands,
        "seed": seed, "seat_swapped": True,
        "agents": {n: agents[n].metadata() for n in names},
        "matrix": matrix,
        "units": "game utility per hand (seat-swap averaged); 95% CI "
                 "from a normal approximation",
    }


__all__ = ["play_episode", "play_match", "round_robin", "MatchResult"]
