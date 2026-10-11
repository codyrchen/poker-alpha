"""Active compute allocation for external-sampling MCCFR (Phase II, Program 2).

Research question: under a fixed compute budget (node traversals / wall
time), does steering sampled traversals toward strategically *unresolved*
regions converge faster than standard uniform chance sampling?

Mechanism: prioritized root-chance sampling with importance correction.
Standard ES-MCCFR samples the root chance outcome (the deal) from its natural
distribution ``p``. The prioritized solver samples it from a mixture

    q_c = (1 - eps) * p_c + eps * score_c / sum(score)

and multiplies every regret and average-strategy increment made under that
outcome by ``p_c / q_c``. Each increment therefore has the same expectation
as under natural sampling (importance sampling), so the estimator remains
unbiased; only the variance / allocation of samples changes. ``eps`` bounds
the importance weights by ``1 / (1 - eps)``.

Priority signals (``score_c`` for root outcome ``c``; recomputed every
``refresh`` iterations from the solver's own tables, except the oracle):

``positive_regret``  sum over information sets reachable under ``c`` of the
                     cumulative positive regret (unvisited sets contribute 0).
``scarcity``         mean over those information sets of ``1 / (1 + visits)``.
``entropy``          mean current-strategy entropy over those sets (nats).
``random_fixed``     a fixed random non-uniform score (control: reweighting
                     without information).
``oracle_br_gain``   exact per-outcome best-response gain of the current
                     average strategy (uses the exact evaluator; an upper
                     bound reference, not a scalable method).

The mapping outcome -> reachable information-set keys is built once by a
full subtree walk, so this applies only to games whose root chance node and
subtrees can be enumerated (Kuhn, Leduc, reduced Hold'em), which is where
exact exploitability is available anyway.

Related literature: targeted / prioritized sampling in MCCFR has been
explored by Lanctot et al. (2009, "outcome sampling" with epsilon-greedy
exploration), by Gibson et al. (2012, probing / variance reduction) and by
"Targeted CFR" style schemes; this module measures the simplest
importance-corrected version under equal-compute accounting and makes no
novelty claim.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..games.base import Game, State
from ..solvers.cfr import strategy_dot
from ..solvers.mccfr import MCCFRSolver

SIGNALS = ("uniform", "positive_regret", "scarcity", "entropy", "random_fixed", "oracle_br_gain")


class CountingGame(Game):
    """Wraps a game and counts node visits (``is_terminal`` calls), which the
    repository's solvers call exactly once per visited node."""

    def __init__(self, inner: Game) -> None:
        self.inner = inner
        self.nodes = 0

    def signature(self):
        return self.inner.signature()

    def root(self):
        return self.inner.root()

    def is_chance(self, s):
        return self.inner.is_chance(s)

    def chance_outcomes(self, s):
        return self.inner.chance_outcomes(s)

    def sample_chance(self, s, rng):
        return self.inner.sample_chance(s, rng)

    def is_terminal(self, s):
        self.nodes += 1
        return self.inner.is_terminal(s)

    def utility(self, s):
        return self.inner.utility(s)

    def current_player(self, s):
        return self.inner.current_player(s)

    def infoset_key(self, s):
        return self.inner.infoset_key(s)

    def legal_actions(self, s):
        return self.inner.legal_actions(s)

    def next_state(self, s, a):
        return self.inner.next_state(s, a)


def outcome_infosets(game: Game) -> List[List[str]]:
    """For each root chance outcome, the sorted list of information-set keys
    reachable in its subtree (both players)."""
    out = []
    for _, s in game.chance_outcomes(game.root()):
        keys = set()
        stack = [s]
        while stack:
            st = stack.pop()
            if game.is_terminal(st):
                continue
            if game.is_chance(st):
                stack.extend(c for _, c in game.chance_outcomes(st))
                continue
            keys.add(game.infoset_key(st))
            stack.extend(game.next_state(st, a) for a in game.legal_actions(st))
        out.append(sorted(keys))
    return out


class PrioritizedMCCFR(MCCFRSolver):
    """ES-MCCFR with importance-corrected prioritized root-chance sampling."""

    def __init__(self, game: Game, seed: int = 0, signal: str = "uniform", eps: float = 0.5,
                 refresh: int = 50, exact_game: Optional[Game] = None) -> None:
        super().__init__(game, seed)
        if signal not in SIGNALS:
            raise ValueError(f"unknown signal {signal!r}")
        self.signal = signal
        self.eps = float(eps)
        self.refresh = int(refresh)
        self.exact_game = exact_game or game
        root = game.root()
        self._root = root
        if not game.is_chance(root):
            raise ValueError("prioritized sampling needs a root chance node")
        self.outcomes = game.chance_outcomes(root)
        self.p = np.array([p for p, _ in self.outcomes], dtype=np.float64)
        self.p /= self.p.sum()
        self.q = self.p.copy()
        # built on the exact (uncounted) game so the one-off tree walk is not
        # charged to the training budget of a CountingGame
        self.keys_by_outcome = outcome_infosets(self.exact_game)
        self.visits: Dict[str, int] = defaultdict(int)
        self.outcome_visits = np.zeros(len(self.outcomes), dtype=np.int64)
        self._fixed = np.random.default_rng(seed + 12345).random(len(self.outcomes)) if signal == "random_fixed" else None
        self._weight = 1.0
        self.max_weight = 1.0

    # -- priorities -------------------------------------------------------
    def _scores(self) -> Optional[np.ndarray]:
        if self.signal == "uniform":
            return None
        if self.signal == "random_fixed":
            return self._fixed
        n = len(self.outcomes)
        scores = np.zeros(n)
        if self.signal == "oracle_br_gain":
            from ..solvers.evaluation import best_response, expected_value
            strat = self.average_strategy()
            # per-outcome contribution of both players' BR gain, computed by
            # restricting the game to each root outcome via exact walks
            for u in (0, 1):
                _, pol = best_response(self.exact_game, strat, u)
                for ci, (pc, s) in enumerate(self.outcomes):
                    scores[ci] += pc * (self._subtree_br(s, u, strat, pol) - self._subtree_profile(s, u, strat))
            return np.maximum(scores, 0.0)
        for ci, keys in enumerate(self.keys_by_outcome):
            tot = 0.0
            for k in keys:
                node = self.infosets.get(k)
                if self.signal == "positive_regret":
                    if node is not None:
                        tot += float(np.maximum(node.regret_sum, 0.0).sum())
                elif self.signal == "scarcity":
                    tot += 1.0 / (1.0 + self.visits.get(k, 0))
                elif self.signal == "entropy":
                    if node is None:
                        tot += np.log(2.0)
                    else:
                        s = node.current_strategy()
                        nz = s[s > 0]
                        tot += float(-(nz * np.log(nz)).sum())
            scores[ci] = tot / (len(keys) if self.signal in ("scarcity", "entropy") else 1.0)
        return scores

    def _subtree_profile(self, s, u, strat):
        from ..solvers.evaluation import _probs_at
        g = self.exact_game
        if g.is_terminal(s):
            x = g.utility(s)
            return x if u == 0 else -x
        if g.is_chance(s):
            return sum(p * self._subtree_profile(c, u, strat) for p, c in g.chance_outcomes(s))
        probs = _probs_at(g, s, strat)
        return sum(probs[a] * self._subtree_profile(g.next_state(s, a), u, strat) for a in g.legal_actions(s))

    def _subtree_br(self, s, u, strat, pol):
        from ..solvers.evaluation import _subtree_value
        return _subtree_value(self.exact_game, s, u, strat, pol)

    def _refresh_q(self) -> None:
        scores = self._scores()
        if scores is None or scores.sum() <= 0.0:
            self.q = self.p.copy()
        else:
            self.q = (1.0 - self.eps) * self.p + self.eps * scores / scores.sum()
            self.q /= self.q.sum()

    # -- traversal with importance weight ----------------------------------
    def _traverse(self, state: State, update_player: int) -> float:
        game = self.game
        if game.is_terminal(state):
            u0 = game.utility(state)
            return u0 if update_player == 0 else -u0
        if game.is_chance(state):
            return self._traverse(game.sample_chance(state, self.rng), update_player)
        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = self._get_infoset(key, actions)
        strategy = node.current_strategy()
        if player != update_player:
            node.strategy_sum += self._weight * strategy
            idx = self._sample(strategy)
            return self._traverse(game.next_state(state, actions[idx]), update_player)
        self.visits[key] += 1
        child_values = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            child_values[i] = self._traverse(game.next_state(state, action), update_player)
        node_value = strategy_dot(strategy, child_values)
        node.regret_sum += self._weight * (child_values - node_value)
        return node_value

    def _traverse_root(self, update_player: int) -> float:
        # visit the root chance node exactly as the base traversal does
        # (keeps node accounting identical to MCCFRSolver)
        self.game.is_terminal(self._root)
        ci = int(self.rng.choice(len(self.q), p=self.q))
        self.outcome_visits[ci] += 1
        self._weight = float(self.p[ci] / self.q[ci])
        self.max_weight = max(self.max_weight, self._weight)
        return self._traverse(self.outcomes[ci][1], update_player)

    def iterate(self) -> float:
        if self.signal != "uniform" and self.iterations % self.refresh == 0:
            self._refresh_q()
        value = self._traverse_root(0)
        self._traverse_root(1)
        self.iterations += 1
        return value
