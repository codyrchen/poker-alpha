"""Strategy evaluation: expected value, best response, and exploitability.

These are the yardsticks for "how good is this strategy?"

* :func:`expected_value` -- the game value to player 0 when both players follow
  a given strategy profile.
* :func:`best_response_value` -- the most a player can win by best-responding to
  a fixed opponent, *respecting information sets* (an imperfect-information best
  response, not a cheating perfect-information one).
* :func:`exploitability` -- how far a profile is from Nash equilibrium: the
  average of both players' best-response gains. It is zero exactly at
  equilibrium and is the primary convergence metric for CFR.

The best response is computed with a counterfactual, per-information-set
argmax. Because a best-responding player must choose one action per information
set (not per history), information sets are resolved from the deepest first so
that, when an information set is decided, every deeper decision it depends on is
already fixed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..games.base import Game, State

Strategy = Dict[str, Dict[str, float]]


def expected_value(game: Game, strategy: Strategy) -> float:
    """Player-0 game value when both players follow ``strategy``."""

    def walk(state: State) -> float:
        if game.is_terminal(state):
            return game.utility(state)
        if game.is_chance(state):
            return sum(p * walk(s) for p, s in game.chance_outcomes(state))
        probs = strategy[game.infoset_key(state)]
        return sum(
            probs[a] * walk(game.next_state(state, a))
            for a in game.legal_actions(state)
        )

    return walk(game.root())


def _subtree_value(game: Game, state: State, br_player: int,
                   strategy: Strategy, br_policy: Dict[str, str]) -> float:
    """Value to ``br_player`` when they follow ``br_policy`` and the opponent
    follows ``strategy`` (chance averaged)."""
    if game.is_terminal(state):
        u0 = game.utility(state)
        return u0 if br_player == 0 else -u0
    if game.is_chance(state):
        return sum(
            p * _subtree_value(game, s, br_player, strategy, br_policy)
            for p, s in game.chance_outcomes(state)
        )
    key = game.infoset_key(state)
    if game.current_player(state) == br_player:
        action = br_policy[key]
        return _subtree_value(game, game.next_state(state, action), br_player,
                              strategy, br_policy)
    probs = strategy[key]
    return sum(
        probs[a] * _subtree_value(game, game.next_state(state, a), br_player,
                                  strategy, br_policy)
        for a in game.legal_actions(state)
    )


def best_response_value(game: Game, strategy: Strategy, br_player: int) -> float:
    """Value to ``br_player`` of best-responding to ``strategy`` (the opponent).

    Respects information sets: the returned value is the true (imperfect-
    information) best-response value, so it is a valid input to exploitability.
    """
    # Group br_player's decision states by information set, recording the
    # counterfactual reach (opponent x chance) of each and its depth.
    groups: Dict[str, List[Tuple[State, float]]] = defaultdict(list)
    depth: Dict[str, int] = {}

    def enumerate_states(state: State, reach: float, d: int) -> None:
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for p, s in game.chance_outcomes(state):
                enumerate_states(s, reach * p, d + 1)
            return
        key = game.infoset_key(state)
        if game.current_player(state) == br_player:
            groups[key].append((state, reach))
            depth[key] = max(depth.get(key, 0), d)
            for a in game.legal_actions(state):
                # br_player's own reach is excluded from the counterfactual reach.
                enumerate_states(game.next_state(state, a), reach, d + 1)
        else:
            probs = strategy[key]
            for a in game.legal_actions(state):
                enumerate_states(game.next_state(state, a), reach * probs[a], d + 1)

    enumerate_states(game.root(), 1.0, 0)

    # Decide each information set deepest-first so dependencies are already fixed.
    br_policy: Dict[str, str] = {}
    for key in sorted(groups, key=lambda k: depth[k], reverse=True):
        states = groups[key]
        actions = game.legal_actions(states[0][0])
        best_action, best_value = actions[0], float("-inf")
        for action in actions:
            value = 0.0
            for state, reach in states:
                value += reach * _subtree_value(
                    game, game.next_state(state, action), br_player,
                    strategy, br_policy)
            if value > best_value:
                best_value, best_action = value, action
        br_policy[key] = best_action

    return _subtree_value(game, game.root(), br_player, strategy, br_policy)


def exploitability(game: Game, strategy: Strategy) -> float:
    """Average best-response gain over both players; zero at Nash equilibrium."""
    br0 = best_response_value(game, strategy, br_player=0)
    br1 = best_response_value(game, strategy, br_player=1)
    return (br0 + br1) / 2.0
