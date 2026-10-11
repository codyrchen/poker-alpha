"""Generic game-complexity inspector (Phase II, Program 14D).

Walks the full tree of any :class:`~poker_alpha.games.base.Game` whose chance
outcomes can be enumerated and counts histories (nodes), chance / decision /
terminal nodes, information sets per player, maximum depth, branching
statistics and checks the interface contracts every solver in this
repository relies on:

* chance probabilities are positive and sum to 1 at every chance node;
* every decision node has at least one legal action, and all histories in
  one information set share the same acting player and action list
  (otherwise per-infoset regret tables are ill-defined);
* information-set keys of the two players never coincide (required by the
  seat-swapped best-response agent);
* terminal utilities are finite.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict

from .base import Game


def inspect_game(game: Game, max_nodes: int = 5_000_000) -> Dict[str, object]:
    stats = Counter()
    branching = Counter()
    infoset_actions: Dict[str, tuple] = {}
    infoset_player: Dict[str, int] = {}
    infoset_histories = Counter()
    problems = []
    max_depth = 0
    stack = [(game.root(), 0)]
    while stack:
        s, d = stack.pop()
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes:
            problems.append(f"truncated at {max_nodes} nodes")
            break
        max_depth = max(max_depth, d)
        if game.is_terminal(s):
            stats["terminal"] += 1
            u = game.utility(s)
            if not math.isfinite(u):
                problems.append("non-finite utility")
            continue
        if game.is_chance(s):
            stats["chance"] += 1
            outs = game.chance_outcomes(s)
            tot = sum(p for p, _ in outs)
            if abs(tot - 1.0) > 1e-9 or any(p <= 0 for p, _ in outs):
                problems.append(f"bad chance distribution (sum {tot})")
            branching[("chance", len(outs))] += 1
            stack.extend((c, d + 1) for _, c in outs)
            continue
        stats["decision"] += 1
        p = game.current_player(s)
        k = game.infoset_key(s)
        acts = tuple(game.legal_actions(s))
        if not acts:
            problems.append(f"no legal actions at {k}")
            continue
        if k in infoset_actions:
            if infoset_actions[k] != acts:
                problems.append(f"infoset {k} has inconsistent action sets")
            if infoset_player[k] != p:
                problems.append(f"infoset {k} is shared by both players")
        infoset_actions[k] = acts
        infoset_player[k] = p
        infoset_histories[k] += 1
        branching[("decision", len(acts))] += 1
        stack.extend((game.next_state(s, a), d + 1) for a in acts)
    per_player = Counter(infoset_player.values())
    dec_b = [n for (kind, n), c in branching.items() if kind == "decision" for _ in range(c)]
    return {"game": game.signature(), "nodes": stats["nodes"], "chance_nodes": stats["chance"],
            "decision_nodes": stats["decision"], "terminal_nodes": stats["terminal"],
            "infosets": len(infoset_actions), "infosets_by_player": {str(k): v for k, v in sorted(per_player.items())},
            "max_depth": max_depth,
            "decision_branching_mean": (sum(dec_b) / len(dec_b)) if dec_b else 0.0,
            "decision_branching_max": max(dec_b) if dec_b else 0,
            "histories_per_infoset_mean": (sum(infoset_histories.values()) / len(infoset_histories)) if infoset_histories else 0.0,
            "sequence_form_size": sum(len(a) for a in infoset_actions.values()),
            "branching": {f"{k[0]}:{k[1]}": v for k, v in sorted(branching.items())},
            "problems": sorted(set(problems))[:20], "ok": not problems}
