"""Exact, Hold'em-shaped validation games (Phase 33).

Small enough for exact CFR / CFR+ / best response, but keeping the
structure that matters in heads-up Hold'em: private two-card hands, blind
asymmetry (button = small blind acts first preflop), fold / call / raise /
all-in with stack- and pot-based chip utilities, and information sets that
contain only the actor's own cards plus public actions (perfect recall).

``ReducedPreflopGame``
    Preflop-only: hands dealt from a reduced deck (default A K Q J T in two
    suits = 10 cards, 45 hands). After the betting the hand goes to
    showdown with **expected** pot share: each player's equity on a random
    5-card board from the rest of the real 52-card deck. Equities are
    computed once (seeded Monte Carlo, ``equity_samples`` boards per hand
    pair) and become the game's definition, so "exact" below is exact for
    this defined game.

``RiverSubgame``
    A fixed 52-card river board (no further chance), explicit weighted
    ranges for both players, a starting pot and stacks, and a bet menu.
    The out-of-position player (BB, player 0 here) acts first. Showdown is
    exact with the 7-card evaluator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..poker.cards import card_str, codes
from ..poker.evaluator import evaluate_best_codes
from .base import Game


# --------------------------------------------------------------------------
# Preflop-only reduced game
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PFState:
    holes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    history: Tuple[str, ...]


class ReducedPreflopGame(Game):
    """See module docstring. Player 0 = button / small blind."""

    SB, BB = 0.5, 1.0

    def __init__(self, ranks: str = "AKQJT", suits: str = "sh", stack: float = 10.0,
                 open_to: float = 2.5, equity_samples: int = 4000, seed: int = 0) -> None:
        self.ranks, self.suits = ranks, suits
        self.stack = float(stack)
        self.open_to = float(open_to)
        self.equity_samples, self.seed = equity_samples, seed
        self.deck = tuple(sorted(codes([r + s for r in ranks for s in suits])))
        self.hands = tuple(itertools.combinations(self.deck, 2))
        self._deals = [(h0, h1) for h0 in self.hands for h1 in self.hands if not set(h0) & set(h1)]
        self._equity = self._equity_table()

    def signature(self) -> str:
        return (f"ReducedPreflopGame:v1:ranks={self.ranks}:suits={self.suits}:stack={self.stack:g}"
                f":open={self.open_to:g}:eq={self.equity_samples}:seed={self.seed}")

    def _equity_table(self) -> Dict[Tuple, float]:
        rng = np.random.default_rng(self.seed)
        table = {}
        for h0, h1 in self._deals:
            if (h1, h0) in table:
                table[(h0, h1)] = 1.0 - table[(h1, h0)]
                continue
            live = [c for c in range(52) if c not in h0 and c not in h1]
            win = 0.0
            for _ in range(self.equity_samples):
                b = list(rng.choice(live, size=5, replace=False))
                v0 = evaluate_best_codes(list(h0) + b)
                v1 = evaluate_best_codes(list(h1) + b)
                win += 1.0 if v0 > v1 else (0.5 if v0 == v1 else 0.0)
            table[(h0, h1)] = win / self.equity_samples
        return table

    # -- betting --------------------------------------------------------
    def _contrib(self, history):
        c = [self.SB, self.BB]
        level = self.BB
        p = 0
        for a in history:
            if a == "c":
                c[p] = min(level, self.stack)
            elif a == "r":
                level = self.open_to
                c[p] = level
            elif a == "a":
                level = self.stack
                c[p] = level
            p = 1 - p
        return c, level, p

    def root(self):
        return PFState(None, ())

    def is_chance(self, s):
        return s.holes is None

    def chance_outcomes(self, s):
        p = 1.0 / len(self._deals)
        return [(p, PFState(d, ())) for d in self._deals]

    def is_terminal(self, s):
        h = s.history
        if not h or s.holes is None:
            return False
        if h[-1] == "f":
            return True
        # a call closes the action, except the button's limp ("c" first)
        return h[-1] == "c" and len(h) >= 2

    def utility(self, s):
        c, _, p = self._contrib(s.history)
        if s.history[-1] == "f":
            folder = 1 - p
            return -c[0] if folder == 0 else c[1]
        eq = self._equity[s.holes]
        pot = c[0] + c[1]
        return eq * pot - c[0]

    def current_player(self, s):
        return len(s.history) % 2

    def infoset_key(self, s):
        p = self.current_player(s)
        h = s.holes[p]
        return f"{p}|{card_str(h[0])}{card_str(h[1])}|{''.join(s.history)}"

    def legal_actions(self, s):
        c, level, p = self._contrib(s.history)
        facing = level > c[p] + 1e-9
        acts = ["f"] if facing else []
        acts.append("c")
        if level < self.open_to - 1e-9:
            acts.append("r")
        if level < self.stack - 1e-9:
            acts.append("a")
        return acts

    def next_state(self, s, a):
        return PFState(s.holes, s.history + (a,))


# --------------------------------------------------------------------------
# Fixed-board river subgame
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RState:
    holes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    history: Tuple[str, ...]


def parse_range(spec: Sequence[str], board: Sequence[int]) -> List[Tuple[Tuple[int, int], float]]:
    """``["AsKs", "QhQd:0.5", ...]`` -> combos (minus board cards) with weights."""
    out = []
    bset = set(board)
    for item in spec:
        cards, _, w = item.partition(":")
        a, b = codes([cards[:2], cards[2:4]])
        if a in bset or b in bset:
            continue
        out.append(((min(a, b), max(a, b)), float(w) if w else 1.0))
    return out


class RiverSubgame(Game):
    """Exact river subgame. Player 0 = out of position (acts first)."""

    def __init__(self, board: Sequence[str], range0: Sequence[str], range1: Sequence[str],
                 pot: float = 10.0, stack: float = 20.0, bets: Sequence[float] = (0.75,),
                 raise_cap: int = 2, name: str = "") -> None:
        self.name = name
        self.board = tuple(codes(board))
        self.r0 = parse_range(range0, self.board)
        self.r1 = parse_range(range1, self.board)
        self.pot, self.stack = float(pot), float(stack)
        self.bets = tuple(bets)
        self.raise_cap = raise_cap
        deals = []
        for h0, w0 in self.r0:
            for h1, w1 in self.r1:
                if not set(h0) & set(h1):
                    deals.append(((h0, h1), w0 * w1))
        tot = sum(w for _, w in deals)
        self._deals = [(d, w / tot) for d, w in deals]
        self._value = {}
        for (h0, h1), _ in self._deals:
            for h in (h0, h1):
                if h not in self._value:
                    self._value[h] = evaluate_best_codes(list(h) + list(self.board))

    def signature(self):
        return f"RiverSubgame:v1:{self.name}:pot={self.pot:g}:stack={self.stack:g}:bets={self.bets}:cap={self.raise_cap}"

    def _tok(self, i):
        return f"b{int(round(self.bets[i] * 100))}"

    def _state(self, history):
        """(contributions this street, level, player to act, raises)."""
        c = [0.0, 0.0]
        level, p, n = 0.0, 0, 0
        for a in history:
            if a == "c":
                c[p] = level
            elif a == "a":
                level = self.stack
                c[p] = level
                n += 1
            elif a.startswith("b"):
                frac = int(a[1:]) / 100.0
                owe = level - c[p]
                pot_now = self.pot + c[0] + c[1]
                level = c[p] + owe + frac * (pot_now + owe)
                c[p] = level
                n += 1
            p = 1 - p
        return c, level, p, n

    def root(self):
        return RState(None, ())

    def is_chance(self, s):
        return s.holes is None

    def chance_outcomes(self, s):
        return [(w, RState(d, ())) for d, w in self._deals]

    def is_terminal(self, s):
        h = s.history
        if not h or s.holes is None:
            return False
        if h[-1] == "f":
            return True
        return h[-1] == "c" and len(h) >= 2

    def utility(self, s):
        c, _, p, _ = self._state(s.history)
        half = self.pot / 2.0
        if s.history[-1] == "f":
            folder = 1 - p
            return -(half + c[0]) if folder == 0 else (half + c[1])
        v0, v1 = self._value[s.holes[0]], self._value[s.holes[1]]
        if v0 > v1:
            return half + c[1]
        if v1 > v0:
            return -(half + c[0])
        return 0.0

    def current_player(self, s):
        return len(s.history) % 2

    def infoset_key(self, s):
        p = self.current_player(s)
        h = s.holes[p]
        return f"{p}|{card_str(h[0])}{card_str(h[1])}|{'.'.join(s.history)}"

    def legal_actions(self, s):
        c, level, p, n = self._state(s.history)
        facing = level > c[p] + 1e-9
        acts = (["f"] if facing else []) + ["c"]
        if n < self.raise_cap and level < self.stack - 1e-9:
            pot_now = self.pot + c[0] + c[1]
            owe = level - c[p]
            for i, frac in enumerate(self.bets):
                to = c[p] + owe + frac * (pot_now + owe)
                if to < self.stack - 1e-9:
                    acts.append(self._tok(i))
            acts.append("a")
        return acts

    def next_state(self, s, a):
        return RState(s.holes, s.history + (a,))
