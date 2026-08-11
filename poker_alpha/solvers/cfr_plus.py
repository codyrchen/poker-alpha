"""CFR+: three changes to vanilla CFR that converge dramatically faster.

CFR+ (Tammelin 2014) modifies vanilla CFR in three ways, all of which matter:

1. **Regret clipping** — cumulative regrets are floored at zero after every
   update (``R <- max(R + r, 0)``). An action that accumulated a mountain of
   negative regret no longer needs many iterations of positive regret to "dig
   out" before regret matching will play it again, so the strategy can react
   quickly when an action becomes good.

2. **Alternating updates** — each iteration performs two traversals, updating
   one player's regrets at a time while the other plays their current strategy.
   Each player therefore responds to the opponent's *already updated* strategy
   within the same iteration. Empirically this is essential: with simultaneous
   updates, clipping + weighted averaging alone yield only a modest speedup.

3. **Linear (weighted) averaging** — iteration ``t`` contributes to the average
   strategy with weight ``t`` instead of 1, damping the early, bad iterates so
   the average is dominated by late, near-equilibrium play.

The traversal below takes an ``update_player`` argument: regret and strategy
sums are accumulated only at that player's information sets, while both players'
action probabilities still shape the values. Utilities remain from player 0's
perspective throughout, converted to the acting player's sign only in the
regret update, exactly as in :class:`~poker_alpha.solvers.cfr.CFRSolver`.

One correctness subtlety: an information set typically contains *several*
histories (in Kuhn, "hold the Jack" spans two possible opponent cards), and the
tree traversal visits the node once per history. The clip must be applied to the
information set's **total** regret for the iteration, not per visit — clipping
between visits would make the update order-dependent and biased. The traversal
therefore buffers per-iteration regret deltas and applies ``max(R + Δ, 0)``
once per information set after the traversal completes.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..games.base import Game, State
from .cfr import CFRSolver


class CFRPlusSolver(CFRSolver):
    """CFR+ over a :class:`~poker_alpha.games.base.Game`.

    Differences from :class:`CFRSolver`: cumulative regrets clipped at zero,
    alternating per-player updates, and linearly weighted strategy averaging.
    """

    def __init__(self, game: Game) -> None:
        super().__init__(game)
        # Per-iteration regret deltas, keyed by infoset; clipped in once per
        # traversal (see module docstring).
        self._pending_regret: Dict[str, np.ndarray] = {}

    def _cfr_plus(self, state: State, update_player: int, reach0: float,
                  reach1: float, reach_chance: float) -> float:
        game = self.game
        if game.is_terminal(state):
            return game.utility(state)
        if game.is_chance(state):
            value = 0.0
            for prob, nxt in game.chance_outcomes(state):
                value += prob * self._cfr_plus(nxt, update_player, reach0,
                                               reach1, reach_chance * prob)
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
                child_values[i] = self._cfr_plus(nxt, update_player,
                                                 reach0 * strategy[i], reach1,
                                                 reach_chance)
            else:
                child_values[i] = self._cfr_plus(nxt, update_player, reach0,
                                                 reach1 * strategy[i],
                                                 reach_chance)
        node_value = float(strategy @ child_values)

        if player == update_player:
            sign = 1.0 if player == 0 else -1.0
            own_reach = reach0 if player == 0 else reach1
            cf_reach = (reach1 if player == 0 else reach0) * reach_chance
            action_regret = sign * (child_values - node_value)
            # Buffer the delta; the clip happens once per infoset per traversal.
            pending = self._pending_regret.get(key)
            if pending is None:
                self._pending_regret[key] = cf_reach * action_regret
            else:
                pending += cf_reach * action_regret
            # Linear averaging: iteration t (1-based) contributes weight t.
            weight = float(self.iterations + 1)
            node.strategy_sum += weight * own_reach * strategy
        return node_value

    def _traverse_for(self, update_player: int) -> float:
        """One full traversal updating one player, then clip regrets once."""
        self._pending_regret.clear()
        value = self._cfr_plus(self.game.root(), update_player, 1.0, 1.0, 1.0)
        for key, delta in self._pending_regret.items():
            node = self.infosets[key]
            # The "+" in CFR+: floor cumulative regret at zero.
            node.regret_sum = np.maximum(node.regret_sum + delta, 0.0)
        return value

    def iterate(self) -> float:
        """One CFR+ iteration = one regret update per player, alternating."""
        value = self._traverse_for(0)
        self._traverse_for(1)
        self.iterations += 1
        return value
