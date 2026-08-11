"""External-sampling Monte Carlo CFR (MCCFR).

Vanilla CFR walks the entire game tree every iteration — cost proportional to
the number of histories, which explodes with game size. MCCFR (Lanctot et al.
2009) replaces the exact traversal with a sampled one whose regret estimates
are unbiased, trading per-iteration accuracy for many more, much cheaper
iterations. Convergence (with high probability) remains O(1/√T), but the
constant per iteration shrinks from "whole tree" to "one trajectory bundle".

*External sampling* is the variant that samples everything **external** to the
updating player — chance outcomes and the opponent's actions are sampled (one
successor each), while all of the updating player's own actions are still
explored. This keeps the updating player's regret estimates well fed while
skipping most of the tree, and is the standard default among MCCFR variants.

Update rules per traversal for player ``i``:

* at ``i``'s nodes: recurse into every action, then accumulate sampled
  counterfactual regret ``r(a) += u(a) − Σ_b σ(b) u(b)``;
* at opponent nodes: sample one action from the opponent's current strategy and
  recurse; accumulate the opponent's *average-strategy* contribution
  ``s(a) += σ(a)`` here (each infoset is reached with probability proportional
  to the opponent's own reach, which makes this unweighted tally correct in
  expectation);
* at chance nodes: sample one outcome by its probability.

Sampling makes results stochastic, so the solver takes an explicit seed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..games.base import Game, State
from .cfr import CFRSolver


class MCCFRSolver(CFRSolver):
    """External-sampling MCCFR over a :class:`~poker_alpha.games.base.Game`."""

    def __init__(self, game: Game, seed: int = 0) -> None:
        super().__init__(game)
        self.rng = np.random.default_rng(seed)

    def _sample(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(len(probs), p=probs))

    def _traverse(self, state: State, update_player: int) -> float:
        """Sampled counterfactual value of ``state`` for ``update_player``."""
        game = self.game
        if game.is_terminal(state):
            u0 = game.utility(state)
            return u0 if update_player == 0 else -u0
        if game.is_chance(state):
            outcomes = game.chance_outcomes(state)
            probs = np.array([p for p, _ in outcomes])
            idx = self._sample(probs / probs.sum())
            return self._traverse(outcomes[idx][1], update_player)

        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = self._get_infoset(key, actions)
        strategy = node.current_strategy()

        if player != update_player:
            # Opponent node: tally average strategy, then sample one action.
            node.strategy_sum += strategy
            idx = self._sample(strategy)
            return self._traverse(game.next_state(state, actions[idx]),
                                  update_player)

        # Updating player's node: explore every action.
        child_values = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            child_values[i] = self._traverse(game.next_state(state, action),
                                             update_player)
        node_value = float(strategy @ child_values)
        node.regret_sum += child_values - node_value
        return node_value

    def iterate(self) -> float:
        """One MCCFR iteration = one sampled traversal per player."""
        value = self._traverse(self.game.root(), 0)
        self._traverse(self.game.root(), 1)
        self.iterations += 1
        return value
