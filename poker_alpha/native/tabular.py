"""Tabular-tree adapter: run native MCCFR directly on exact reduced games.

``build_tabular_tree(game)`` flattens any two-player zero-sum
:class:`~poker_alpha.games.base.Game` whose ONLY chance node is the root
deal (ReducedPreflopGame, RiverSubgame) into the arrays the native
``TabularSolver`` consumes: a betting tree shared by every deal, the
weighted deal list, and a dense per-terminal utility matrix
``u[terminal][h0][h1]`` computed by calling the *Python* game's own
``utility`` — the native solver trains on exactly the game Python defines,
with zero rule duplication.

``NativeTabularMCCFR`` wraps the extension and returns strategies in the
standard ``{infoset_key: {action: prob}}`` form, with keys produced by the
Python game's own ``infoset_key`` — directly comparable to exact CFR+ /
Python MCCFR output and usable with the existing exploitability and
best-response evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TabularTree:
    player: np.ndarray        # int8 per node, -1 = terminal
    child_off: np.ndarray     # uint32, len = nodes + 1
    children: np.ndarray      # uint32, ordered by node then action
    term_idx: np.ndarray      # int32 per node, -1 for decision nodes
    node_actions: List[Tuple[str, ...]]   # tokens per node (() for terminals)
    node_paths: List[Tuple[str, ...]]     # action path from the root per node
    deal_weights: np.ndarray
    deal_h0: np.ndarray
    deal_h1: np.ndarray
    hands0: list
    hands1: list
    util: np.ndarray          # float64 [n_term * n_h0 * n_h1]
    node_keys: List[Dict[int, str]]       # decision node -> own hand id -> key


def build_tabular_tree(game) -> TabularTree:
    root = game.root()
    if not game.is_chance(root):
        raise ValueError("adapter expects a chance root (the deal)")
    outcomes = game.chance_outcomes(root)

    hands0: list = []
    hands1: list = []
    h0_index: Dict = {}
    h1_index: Dict = {}
    deals: List[Tuple[float, int, int]] = []
    pair_state: Dict[Tuple[int, int], object] = {}
    for prob, state in outcomes:
        h0, h1 = state.holes
        i0 = h0_index.setdefault(h0, len(hands0))
        if i0 == len(hands0):
            hands0.append(h0)
        i1 = h1_index.setdefault(h1, len(hands1))
        if i1 == len(hands1):
            hands1.append(h1)
        deals.append((prob, i0, i1))
        pair_state[(i0, i1)] = state
    if len(hands0) > 0xFFFF or len(hands1) > 0xFFFF:
        raise ValueError("too many hands for the tabular adapter")

    # --- betting tree (shared across deals; probed with one deal) ---------
    probe = next(iter(pair_state.values()))
    player: List[int] = []
    node_actions: List[Tuple[str, ...]] = []
    node_paths: List[Tuple[str, ...]] = []
    node_children: List[List[int]] = []
    term_of_node: List[int] = []
    terminals: List[int] = []

    def expand(state, path: Tuple[str, ...]) -> int:
        if game.is_chance(state):
            raise ValueError("non-root chance nodes are not supported")
        node = len(player)
        player.append(-1)
        node_actions.append(())
        node_paths.append(path)
        node_children.append([])
        term_of_node.append(-1)
        if game.is_terminal(state):
            term_of_node[node] = len(terminals)
            terminals.append(node)
            return node
        player[node] = game.current_player(state)
        actions = tuple(game.legal_actions(state))
        node_actions[node] = actions
        node_children[node] = [expand(game.next_state(state, a), path + (a,))
                               for a in actions]
        return node

    expand(probe, ())

    n_nodes = len(player)
    child_off = np.zeros(n_nodes + 1, dtype=np.uint32)
    flat_children: List[int] = []
    for node in range(n_nodes):
        flat_children.extend(node_children[node])
        child_off[node + 1] = len(flat_children)

    # --- utility matrices: the Python game defines every payoff -----------
    n0, n1 = len(hands0), len(hands1)
    util = np.zeros((len(terminals), n0, n1), dtype=np.float64)
    for (i0, i1), state0 in pair_state.items():
        for tnode in terminals:
            s = state0
            for a in node_paths[tnode]:
                s = game.next_state(s, a)
            assert game.is_terminal(s)
            util[term_of_node[tnode], i0, i1] = game.utility(s)

    # --- infoset keys: node x own hand, named by the Python game ----------
    own0_rep = {}
    own1_rep = {}
    for (i0, i1), state in pair_state.items():
        own0_rep.setdefault(i0, state)
        own1_rep.setdefault(i1, state)
    node_keys: List[Dict[int, str]] = []
    for node in range(n_nodes):
        keys: Dict[int, str] = {}
        if player[node] >= 0:
            reps = own0_rep if player[node] == 0 else own1_rep
            for own, state0 in reps.items():
                s = state0
                for a in node_paths[node]:
                    s = game.next_state(s, a)
                keys[own] = game.infoset_key(s)
        node_keys.append(keys)

    return TabularTree(
        player=np.array(player, dtype=np.int8),
        child_off=child_off,
        children=np.array(flat_children, dtype=np.uint32),
        term_idx=np.array(term_of_node, dtype=np.int32),
        node_actions=node_actions,
        node_paths=node_paths,
        deal_weights=np.array([w for w, _, _ in deals], dtype=np.float64),
        deal_h0=np.array([a for _, a, _ in deals], dtype=np.uint16),
        deal_h1=np.array([b for _, _, b in deals], dtype=np.uint16),
        hands0=hands0, hands1=hands1,
        util=np.ascontiguousarray(util.reshape(-1)),
        node_keys=node_keys)


class NativeTabularMCCFR:
    """Native external-sampling MCCFR on a flattened exact game."""

    def __init__(self, game, seed: int = 0, tree: "TabularTree | None" = None) -> None:
        from . import native_module

        mod = native_module()
        if mod is None:
            raise RuntimeError("native backend not installed")
        self.game = game
        self.tree = tree if tree is not None else build_tabular_tree(game)
        t = self.tree
        self._core = mod.TabularSolver(
            player=t.player, child_off=t.child_off, children=t.children,
            term_idx=t.term_idx, deal_weights=t.deal_weights,
            deal_h0=t.deal_h0, deal_h1=t.deal_h1,
            n_h0=len(t.hands0), n_h1=len(t.hands1), util=t.util, seed=seed)

    def train(self, iterations: int) -> None:
        self._core.train(iterations)

    @property
    def iterations(self) -> int:
        return int(self._core.iterations)

    def average_strategy(self) -> Dict[str, Dict[str, float]]:
        return {k: p for k, (p, _) in self.average_strategy_with_visits().items()}

    def average_strategy_with_visits(self) -> Dict[str, Tuple[Dict[str, float], float]]:
        """``{key: (probs, visits)}`` — visits are the non-updating-player
        tallies, the same quantity production confidence tables use."""
        out: Dict[str, Tuple[Dict[str, float], float]] = {}
        t = self.tree
        for node, hand, probs, visits in self._core.average_strategy():
            key = t.node_keys[node][hand]
            out[key] = (dict(zip(t.node_actions[node], probs)), float(visits))
        return out
