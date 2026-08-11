"""Vanilla Counterfactual Regret Minimization (CFR).

CFR is a self-play algorithm that provably converges to a Nash equilibrium of a
two-player zero-sum imperfect-information game. At every information set it keeps
a running total of *counterfactual regret* for each action -- how much better,
in hindsight and weighted by the probability of reaching that information set,
the player would have done by always choosing that action. Each iteration it
plays the *regret-matching* strategy (mix in proportion to positive regret), and
the time-average of those strategies is what converges to equilibrium.

This is the deterministic, full-tree-traversal variant: every iteration walks
the whole game tree, enumerating chance outcomes rather than sampling them. It is
the clearest version to reason about; MCCFR (sampling) comes later for speed.

The traversal returns every value from *player 0's* perspective and folds chance
into the reach probabilities, which keeps chance nodes (including public cards in
larger games) correct without special-casing the caller.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..games.base import Game, State


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Convert regrets to a strategy by normalizing positive regret.

    If no action has positive regret (e.g. at initialization), fall back to a
    uniform strategy.
    """
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0.0:
        return positive / total
    return np.full(len(regrets), 1.0 / len(regrets))


class InfoSet:
    """Per-information-set CFR statistics."""

    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions: List[str]) -> None:
        self.actions = actions
        n = len(actions)
        self.regret_sum = np.zeros(n, dtype=np.float64)
        self.strategy_sum = np.zeros(n, dtype=np.float64)

    def current_strategy(self) -> np.ndarray:
        """Regret-matching strategy for the current iteration."""
        return regret_matching(self.regret_sum)

    def average_strategy(self) -> np.ndarray:
        """Time-averaged strategy -- the quantity that converges to Nash."""
        total = self.strategy_sum.sum()
        if total > 0.0:
            return self.strategy_sum / total
        return np.full(len(self.actions), 1.0 / len(self.actions))


class CFRSolver:
    """Vanilla CFR over a :class:`~poker_alpha.games.base.Game`."""

    def __init__(self, game: Game) -> None:
        self.game = game
        self.infosets: Dict[str, InfoSet] = {}
        self.iterations = 0

    # -- infoset bookkeeping --------------------------------------------

    def _get_infoset(self, key: str, actions: List[str]) -> InfoSet:
        node = self.infosets.get(key)
        if node is None:
            node = InfoSet(actions)
            self.infosets[key] = node
        return node

    # -- core traversal -------------------------------------------------

    def _cfr(self, state: State, reach0: float, reach1: float, reach_chance: float) -> float:
        game = self.game
        if game.is_terminal(state):
            return game.utility(state)
        if game.is_chance(state):
            value = 0.0
            for prob, nxt in game.chance_outcomes(state):
                value += prob * self._cfr(state=nxt, reach0=reach0, reach1=reach1,
                                          reach_chance=reach_chance * prob)
            return value

        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = self._get_infoset(key, actions)
        strategy = node.current_strategy()

        # Child values from player 0's perspective.
        child_values = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            nxt = game.next_state(state, action)
            if player == 0:
                child_values[i] = self._cfr(nxt, reach0 * strategy[i], reach1,
                                            reach_chance)
            else:
                child_values[i] = self._cfr(nxt, reach0, reach1 * strategy[i],
                                            reach_chance)
        node_value = float(strategy @ child_values)

        # Regret update, weighted by the counterfactual reach (everyone but the
        # acting player). Values are converted to the acting player's sign.
        sign = 1.0 if player == 0 else -1.0
        own_reach = reach0 if player == 0 else reach1
        cf_reach = (reach1 if player == 0 else reach0) * reach_chance
        action_regret = sign * (child_values - node_value)
        node.regret_sum += cf_reach * action_regret
        node.strategy_sum += own_reach * strategy
        return node_value

    # -- public API -----------------------------------------------------

    def iterate(self) -> float:
        """Run one CFR iteration; return the current player-0 tree value."""
        value = self._cfr(self.game.root(), 1.0, 1.0, 1.0)
        self.iterations += 1
        return value

    def average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Return the average strategy as ``{infoset_key: {action: prob}}``."""
        out: Dict[str, Dict[str, float]] = {}
        for key, node in self.infosets.items():
            probs = node.average_strategy()
            out[key] = {a: float(p) for a, p in zip(node.actions, probs)}
        return out

    def train(self, iterations: int, snapshot_every: int = 0,
              evaluator=None) -> List[dict]:
        """Train for ``iterations`` iterations.

        If ``snapshot_every`` > 0 and an ``evaluator`` callable is provided, the
        evaluator is called on the current average strategy at those intervals
        and its returned dict (plus the iteration index) is collected into the
        returned history list -- used to trace convergence.
        """
        history: List[dict] = []
        for t in range(1, iterations + 1):
            self.iterate()
            if snapshot_every and evaluator and (t % snapshot_every == 0 or t == 1):
                record = {"iteration": t}
                record.update(evaluator(self.average_strategy()))
                history.append(record)
        return history
