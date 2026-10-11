"""Literature CFR variants for the Phase II algorithm benchmark.

* :class:`DiscountedCFRSolver` — Discounted CFR (Brown & Sandholm, AAAI
  2019). Alternating updates; after iteration ``t`` cumulative positive
  regrets are multiplied by ``t^a / (t^a + 1)``, negative regrets by
  ``t^b / (t^b + 1)`` and the cumulative average-strategy contributions by
  ``(t / (t + 1))^g``. ``DCFR`` is ``(a, b, g) = (1.5, 0, 2)``; ``LCFR``
  (Linear CFR) is ``(1, 1, 1)`` (iteration ``t`` weighted by ``t``);
  ``b = -inf`` zeroes negative regret (CFR+'s floor) and ``a = inf`` leaves
  positive regret undiscounted, so ``(inf, -inf, 1)`` is CFR+.
* :class:`OutcomeSamplingMCCFR` — outcome-sampling MCCFR (Lanctot et al.,
  NeurIPS 2009): one sampled trajectory per update, epsilon-greedy
  exploration for the updating player, importance-weighted regrets and
  stochastically-weighted average strategy.

Both operate on the generic :class:`~poker_alpha.games.base.Game` interface
and are evaluated with the exact :func:`~poker_alpha.solvers.evaluation.exploitability`.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..games.base import Game, State
from ..solvers.cfr import CFRSolver, InfoSet, strategy_dot


class DiscountedCFRSolver(CFRSolver):
    """Discounted CFR family with alternating updates (see module docstring)."""

    PRESETS = {"dcfr": (1.5, 0.0, 2.0), "lcfr": (1.0, 1.0, 1.0), "cfr+": (math.inf, -math.inf, 1.0)}

    def __init__(self, game: Game, alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0) -> None:
        super().__init__(game)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self._pending: Dict[str, np.ndarray] = {}
        self._touched: set = set()

    @classmethod
    def preset(cls, game: Game, name: str) -> "DiscountedCFRSolver":
        return cls(game, *cls.PRESETS[name])

    def _walk(self, state: State, update_player: int, reach0: float, reach1: float,
              reach_chance: float) -> float:
        game = self.game
        if game.is_terminal(state):
            return game.utility(state)
        if game.is_chance(state):
            value = 0.0
            for prob, nxt in game.chance_outcomes(state):
                value += prob * self._walk(nxt, update_player, reach0, reach1, reach_chance * prob)
            return value
        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = self._get_infoset(key, actions)
        strategy = node.current_strategy()
        child_values = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            nxt = game.next_state(state, action)
            if player == 0:
                child_values[i] = self._walk(nxt, update_player, reach0 * strategy[i], reach1, reach_chance)
            else:
                child_values[i] = self._walk(nxt, update_player, reach0, reach1 * strategy[i], reach_chance)
        node_value = strategy_dot(strategy, child_values)
        if player == update_player:
            sign = 1.0 if player == 0 else -1.0
            own_reach = reach0 if player == 0 else reach1
            cf_reach = (reach1 if player == 0 else reach0) * reach_chance
            delta = cf_reach * sign * (child_values - node_value)
            pending = self._pending.get(key)
            if pending is None:
                self._pending[key] = delta
            else:
                pending += delta
            node.strategy_sum += own_reach * strategy
            self._touched.add(key)
        return node_value

    @staticmethod
    def _factor(t: float, power: float) -> float:
        if power == math.inf:
            return 1.0
        if power == -math.inf:
            return 0.0
        tp = t ** power
        return tp / (tp + 1.0)

    def iterate(self) -> float:
        t = float(self.iterations + 1)
        value = 0.0
        for u in (0, 1):
            self._pending.clear()
            v = self._walk(self.game.root(), u, 1.0, 1.0, 1.0)
            if u == 0:
                value = v
            for key, delta in self._pending.items():
                self.infosets[key].regret_sum += delta
        # discount cumulative values at the end of iteration t
        fp, fn, fs = self._factor(t, self.alpha), self._factor(t, self.beta), (t / (t + 1.0)) ** self.gamma
        for node in self.infosets.values():
            r = node.regret_sum
            node.regret_sum = np.where(r > 0.0, r * fp, r * fn)
            node.strategy_sum *= fs
        self.iterations += 1
        return value


class OutcomeSamplingMCCFR(CFRSolver):
    """Outcome-sampling MCCFR with epsilon-greedy exploration (alternating
    update player per traversal)."""

    def __init__(self, game: Game, seed: int = 0, epsilon: float = 0.6) -> None:
        super().__init__(game)
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(epsilon)

    def _sample(self, probs: np.ndarray) -> int:
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]
        return int(np.searchsorted(cdf, self.rng.random(), side="right"))

    def _traverse(self, state: State, u: int, reach_u: float, reach_o: float, sample_prob: float):
        """Returns ``(utility for u at the sampled terminal z, u's own reach
        from this state to z under sigma, sampling probability of the suffix
        from this state to z)``."""
        game = self.game
        if game.is_terminal(state):
            x = game.utility(state)
            return (x if u == 0 else -x), 1.0, 1.0
        if game.is_chance(state):
            return self._traverse(game.sample_chance(state, self.rng), u, reach_u, reach_o, sample_prob)
        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = self._get_infoset(key, actions)
        sigma = node.current_strategy()
        n = len(actions)
        if player == u:
            explore = (1.0 - self.epsilon) * sigma + self.epsilon / n
            idx = self._sample(explore)
            q = float(explore[idx])
            util, tail, suffix_q = self._traverse(game.next_state(state, actions[idx]), u,
                                                  reach_u * sigma[idx], reach_o, sample_prob * q)
            # Lanctot et al. 2009: r(I,a') = u(z) pi_-i(h) / q(z) * (pi_i(h a', z) - pi_i(h, z))
            w = util * reach_o / (sample_prob * q * suffix_q)
            regret = np.full(n, -w * sigma[idx] * tail)
            regret[idx] += w * tail
            node.regret_sum += regret
            # stochastically-weighted averaging: own reach / sampling prob of reaching h
            node.strategy_sum += (reach_u / sample_prob) * sigma
            return util, tail * sigma[idx], q * suffix_q
        idx = self._sample(sigma)
        q = float(sigma[idx])
        util, tail, suffix_q = self._traverse(game.next_state(state, actions[idx]), u, reach_u,
                                              reach_o * q, sample_prob * q)
        return util, tail, q * suffix_q

    def iterate(self) -> float:
        v, _, _ = self._traverse(self.game.root(), 0, 1.0, 1.0, 1.0)
        self._traverse(self.game.root(), 1, 1.0, 1.0, 1.0)
        self.iterations += 1
        return v
