"""Agent adapters over existing PokerAlpha machinery."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional

from ..games.base import Game


class RandomAgent:
    """Uniform random over legal actions — the mandatory baseline."""

    def choose_action(self, observation: str, legal_actions: List[str],
                      rng: random.Random) -> str:
        return rng.choice(legal_actions)

    def reset(self) -> None:
        pass

    def metadata(self) -> Dict[str, str]:
        return {"algorithm": "uniform random", "information": "legal actions",
                "adaptive": "no", "source": "none",
                "limitations": "baseline only"}


class PolicyTableAgent:
    """Samples from a trained average-strategy table (CFR / CFR+ / MCCFR
    output: {infoset_key: {action: probability}}). Unseen infosets fall
    back to uniform and are counted."""

    def __init__(self, strategy: Mapping[str, Mapping[str, float]],
                 name: str = "policy table") -> None:
        self.strategy = strategy
        self.name = name
        self.unseen = 0

    def choose_action(self, observation: str, legal_actions: List[str],
                      rng: random.Random) -> str:
        probs = self.strategy.get(observation)
        if not probs:
            self.unseen += 1
            return rng.choice(legal_actions)
        weights = [max(float(probs.get(a, 0.0)), 0.0) for a in legal_actions]
        total = sum(weights)
        if total <= 0:
            self.unseen += 1
            return rng.choice(legal_actions)
        return rng.choices(legal_actions, weights=weights, k=1)[0]

    def reset(self) -> None:
        self.unseen = 0

    def metadata(self) -> Dict[str, str]:
        return {"algorithm": f"fixed average strategy ({self.name})",
                "information": "own information set only",
                "adaptive": "no", "source": "CFR-family training",
                "limitations": "uniform fallback on unseen infosets"}


class BestResponseAgent:
    """Exact best response to a FIXED known opponent strategy — the
    exploitability ceiling for small games. Computes the best-response
    policy for BOTH seats (so seat-swapped matches are meaningful); this
    requires the game's infoset keys to be globally unique across
    players, which is checked. Small exact games only."""

    def __init__(self, game: Game, opponent_strategy) -> None:
        from ..solvers.evaluation import best_response

        self.values = {}
        self.policy = {}
        for seat in (0, 1):
            v, pol = best_response(game, opponent_strategy, seat)
            overlap = set(pol) & set(self.policy)
            if overlap:
                raise ValueError("infoset keys collide across seats; "
                                 "per-seat best response not representable")
            self.values[seat] = v
            self.policy.update(pol)

    def choose_action(self, observation: str, legal_actions: List[str],
                      rng: random.Random) -> str:
        a = self.policy.get(observation)
        return a if a in legal_actions else legal_actions[0]

    def reset(self) -> None:
        pass

    def metadata(self) -> Dict[str, str]:
        return {"algorithm": "exact best response (oracle), both seats",
                "information": "opponent's FULL strategy (not a fair player)",
                "adaptive": "no", "source": "exact tree evaluation",
                "limitations": "upper bound / diagnostic only"}


class ScriptedKuhnAgent:
    """Deterministic Kuhn archetypes with fully documented rules.

    tight:      bet/call only with the King; otherwise pass (fold to bets).
    aggressive: always bet; always call.
    """

    RULES = {"tight": "bet/call only with K; otherwise pass/fold",
             "aggressive": "always bet / always call"}

    def __init__(self, style: str) -> None:
        if style not in self.RULES:
            raise ValueError(f"unknown style {style!r}")
        self.style = style

    def choose_action(self, observation: str, legal_actions: List[str],
                      rng: random.Random) -> str:
        card = int(observation[0])          # Kuhn key = "<card><history>"
        if self.style == "aggressive":
            return "b" if "b" in legal_actions else legal_actions[0]
        return ("b" if card == 2 else "p") \
            if "b" in legal_actions else legal_actions[0]

    def reset(self) -> None:
        pass

    def metadata(self) -> Dict[str, str]:
        return {"algorithm": f"scripted ({self.style}): {self.RULES[self.style]}",
                "information": "own card only", "adaptive": "no",
                "source": "hand-written rule", "limitations": "deterministic"}


def kuhn_policy_agent(iterations: int = 20000) -> PolicyTableAgent:
    """CFR-trained Kuhn policy (deterministic vanilla CFR)."""
    from ..games.kuhn import KuhnPoker
    from ..solvers.cfr import CFRSolver

    solver = CFRSolver(KuhnPoker())
    solver.train(iterations)
    return PolicyTableAgent(solver.average_strategy(),
                            name=f"Kuhn CFR {iterations:,} iters")


__all__ = ["RandomAgent", "PolicyTableAgent", "BestResponseAgent",
           "ScriptedKuhnAgent", "kuhn_policy_agent"]
