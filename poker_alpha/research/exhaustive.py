"""Exhaustive reachable-state enumeration of a reduced Hold'em configuration
(Phase II, Program 9).

The real :class:`~poker_alpha.games.holdem.HoldemGame` cannot enumerate its
52-card chance nodes. This module enumerates every reachable state of the
*same rules engine* when the cards come from a small deck (>= 9 cards so a
full runout exists) with small stacks and a reduced bet menu, and checks a
set of rule invariants at every state. Nothing about the engine is mocked:
the enumerator only replaces the chance distribution (which the engine
exposes via ``sample_chance``'s structure) by exhaustive enumeration over
the reduced deck.

Invariants checked at every non-chance state
--------------------------------------------
* contrib cache equals the replayed contributions (chip conservation)
* 0 <= contrib[i] <= starting stack (non-negative stacks)
* pot == contrib[0] + contrib[1]
* board size matches the street (street monotonicity: 0/3/4/5 cards)
* no duplicate cards among holes and board
* decision nodes: current player in {0, 1}; legal actions non-empty; 'c'
  always legal; 'f' legal iff facing chips; no bet/raise offered once the
  opponent is all-in; bets strictly below the stack ('a' covers the top)
* a folded state is terminal; a matched all-in closes betting
* terminal utility is finite and bounded by the contributions (zero-sum
  payoff feasibility)
* no path exceeds the depth bound (no cycles / deadlocks: every DFS path
  ends in a terminal)
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..games.holdem import _BOARD_SIZE, HoldemGame, HoldemState


@dataclass
class EnumerationReport:
    states: int = 0
    chance_nodes: int = 0
    decision_nodes: int = 0
    terminal_nodes: int = 0
    fold_terminals: int = 0
    showdown_terminals: int = 0
    max_depth: int = 0
    infosets: int = 0
    violations: List[dict] = None
    branching: Dict[int, int] = None
    street_decisions: Dict[int, int] = None
    seconds: float = 0.0
    truncated: bool = False

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["violations"] = self.violations[:50]
        d["violation_count"] = len(self.violations)
        return d


class ReducedDeckEnumerator:
    """Exhaustive DFS over a HoldemGame with cards restricted to ``deck``."""

    def __init__(self, game: HoldemGame, deck: Sequence[int], max_states: Optional[int] = None,
                 encoder=None, depth_bound: int = 64) -> None:
        if len(deck) < 9:
            raise ValueError("deck needs at least 9 cards for a full runout")
        self.game = game
        self.deck = tuple(sorted(set(int(c) for c in deck)))
        self.max_states = max_states
        self.encoder = encoder
        self.depth_bound = depth_bound

    # -- chance ----------------------------------------------------------
    def chance_outcomes(self, state: HoldemState) -> List[HoldemState]:
        if state.holes is None:
            out = []
            for h0 in itertools.combinations(self.deck, 2):
                rest = [c for c in self.deck if c not in h0]
                for h1 in itertools.combinations(rest, 2):
                    out.append(replace(state, holes=(h0, h1)))
            return out
        dead = set(state.board) | {c for h in state.holes for c in h}
        live = [c for c in self.deck if c not in dead]
        need = _BOARD_SIZE[state.street + 1] - len(state.board)
        return [replace(state, board=state.board + cards, streets=state.streets + ("",))
                for cards in itertools.combinations(live, need)]

    # -- invariants ------------------------------------------------------
    def check(self, s: HoldemState, depth: int) -> List[str]:
        g = self.game
        bad = []
        street_paid, total, to_act, n_raises = g._replay(s)
        if any(abs(total[i] - s.contrib[i]) > 1e-9 for i in (0, 1)):
            bad.append("contrib cache != replayed contributions")
        for i in (0, 1):
            if s.contrib[i] < -1e-12 or s.contrib[i] > g.starting_stack + 1e-9:
                bad.append(f"contrib[{i}] out of range")
        if abs(s.pot - (s.contrib[0] + s.contrib[1])) > 1e-9:
            bad.append("pot != sum of contributions")
        if len(s.board) != _BOARD_SIZE[s.street]:
            bad.append("board size does not match street")
        cards = [c for h in s.holes for c in h] + list(s.board)
        if len(set(cards)) != len(cards):
            bad.append("duplicate card")
        if depth > self.depth_bound:
            bad.append("depth bound exceeded")
        if g.is_terminal(s):
            u = g.utility(s)
            if not math.isfinite(u):
                bad.append("non-finite utility")
            if u > s.contrib[1] + 1e-9 or u < -s.contrib[0] - 1e-9:
                bad.append("utility exceeds contributions")
            if s.folded == -1 and len(s.board) != 5:
                bad.append("showdown before the river")
            return bad
        if s.folded != -1:
            bad.append("folded state is not terminal")
        if g.is_chance(s):
            return bad
        p = g.current_player(s)
        if p not in (0, 1):
            bad.append("invalid current player")
        acts = g.legal_actions(s)
        if not acts:
            bad.append("no legal actions at decision node")
            return bad
        owe = street_paid[1 - p] - street_paid[p]
        if "c" not in acts:
            bad.append("'c' missing")
        if (owe > 1e-9) != ("f" in acts):
            bad.append("fold legality inconsistent with facing chips")
        opp_stack = g.starting_stack - total[1 - p]
        my_stack = g.starting_stack - total[p]
        raises = [a for a in acts if a not in ("f", "c")]
        if opp_stack <= 1e-9 and raises:
            bad.append("raise offered against an all-in opponent")
        if my_stack <= owe + 1e-9 and raises:
            bad.append("raise offered without chips beyond the call")
        if s.all_in and abs(s.contrib[0] - s.contrib[1]) < 1e-9:
            bad.append("matched all-in but betting continues")
        for a in raises:
            nxt = g.next_state(s, a)
            if nxt.contrib[p] > g.starting_stack + 1e-9:
                bad.append(f"action {a} exceeds stack")
            if a != "a" and nxt.contrib[p] >= g.starting_stack - 1e-9:
                bad.append(f"sized bet {a} is not strictly below the stack")
        return bad

    # -- enumeration -----------------------------------------------------
    def run(self) -> EnumerationReport:
        g = self.game
        rep = EnumerationReport(violations=[], branching=Counter(), street_decisions=Counter())
        keys = set()
        t0 = time.perf_counter()
        stack: List[Tuple[HoldemState, int]] = [(g.root(), 0)]
        while stack:
            s, depth = stack.pop()
            rep.states += 1
            rep.max_depth = max(rep.max_depth, depth)
            if self.max_states and rep.states > self.max_states:
                rep.truncated = True
                break
            if g.is_chance(s):
                rep.chance_nodes += 1
                if s.holes is not None:
                    for v in self.check(s, depth):
                        rep.violations.append({"state": _describe(s), "violation": v})
                for c in self.chance_outcomes(s):
                    stack.append((c, depth + 1))
                continue
            for v in self.check(s, depth):
                rep.violations.append({"state": _describe(s), "violation": v})
            if g.is_terminal(s):
                rep.terminal_nodes += 1
                if s.folded != -1:
                    rep.fold_terminals += 1
                else:
                    rep.showdown_terminals += 1
                continue
            rep.decision_nodes += 1
            rep.street_decisions[s.street] += 1
            acts = g.legal_actions(s)
            rep.branching[len(acts)] += 1
            keys.add(self.encoder.encode(g, s) if self.encoder is not None else g.infoset_key(s))
            for a in acts:
                stack.append((g.next_state(s, a), depth + 1))
        rep.infosets = len(keys)
        rep.seconds = time.perf_counter() - t0
        rep.branching = dict(sorted(rep.branching.items()))
        rep.street_decisions = dict(sorted(rep.street_decisions.items()))
        return rep


def _describe(s: HoldemState) -> dict:
    return {"holes": s.holes, "board": list(s.board), "streets": list(s.streets),
            "contrib": list(s.contrib), "folded": s.folded, "all_in": s.all_in}


# --------------------------------------------------------------------------
# Metamorphic transformations
# --------------------------------------------------------------------------

def permute_suits(state: HoldemState, perm: Sequence[int]) -> HoldemState:
    """Apply a permutation of the four suits (card = rank * 4 + suit)."""
    def f(c: int) -> int:
        return (c // 4) * 4 + perm[c % 4]
    holes = None if state.holes is None else tuple(tuple(sorted(f(c) for c in h)) for h in state.holes)
    return replace(state, holes=holes, board=tuple(f(c) for c in state.board))


def swap_hole_order(state: HoldemState) -> HoldemState:
    holes = None if state.holes is None else tuple((h[1], h[0]) for h in state.holes)
    return replace(state, holes=holes)


def replay_from_root(game: HoldemGame, state: HoldemState) -> HoldemState:
    """Rebuild ``state`` by applying its action history from the root (with
    the same cards); the result must equal ``state`` field by field."""
    s = replace(game.root(), holes=state.holes)
    for si, actions in enumerate(state.streets):
        if si > 0:
            s = replace(s, board=state.board[:_BOARD_SIZE[si]], streets=s.streets + ("",))
        from ..games.holdem import _tokens
        for tok in _tokens(actions):
            s = game.next_state(s, tok)
    if len(state.board) > len(s.board):
        s = replace(s, board=state.board)
    return s


def state_to_dict(s: HoldemState) -> dict:
    return {"holes": None if s.holes is None else [list(h) for h in s.holes], "board": list(s.board),
            "streets": list(s.streets), "contrib": list(s.contrib), "folded": s.folded, "all_in": s.all_in}


def state_from_dict(d: dict) -> HoldemState:
    return HoldemState(holes=None if d["holes"] is None else tuple(tuple(int(c) for c in h) for h in d["holes"]),
                       board=tuple(int(c) for c in d["board"]), streets=tuple(d["streets"]),
                       contrib=(float(d["contrib"][0]), float(d["contrib"][1])), folded=int(d["folded"]),
                       all_in=bool(d["all_in"]))


def sample_states(enumerator: ReducedDeckEnumerator, n: int, seed: int = 0) -> List[HoldemState]:
    """Reservoir-sample ``n`` reachable non-chance states (for metamorphic tests)."""
    import random
    rng = random.Random(seed)
    g = enumerator.game
    out: List[HoldemState] = []
    seen = 0
    stack = [g.root()]
    while stack:
        s = stack.pop()
        if g.is_chance(s):
            stack.extend(enumerator.chance_outcomes(s))
            continue
        seen += 1
        if len(out) < n:
            out.append(s)
        else:
            j = rng.randrange(seen)
            if j < n:
                out[j] = s
        if not g.is_terminal(s):
            stack.extend(g.next_state(s, a) for a in g.legal_actions(s))
        if seen > 200000:
            break
    return out
