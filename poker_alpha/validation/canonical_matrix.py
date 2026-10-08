"""Broad canonical-state matrix for the locked solver config (Phase 29).

Spots span preflop hand classes x situations (BTN unopened, BB vs limp /
33 / 75 / 150% raise / all-in) and, postflop, board textures x hand
categories x preflop lines (limped, 75% and 150% raise: different SPRs) x
decision nodes (first to act, after a check, facing 33 / 75 / 150% bets) on
the flop, plus turn and river lines. Representative hole cards per (board,
category) are chosen deterministically from the cheap card features, so the
matrix is reproducible and independent of any trained strategy.

Policies are read through a ``lookup(key) -> (probs by token, visits) | None``
callable, so the same matrix evaluates a live solver or an exported
strategy artifact. Unvisited infosets are reported as ``UNVISITED``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..abstraction.features import card_features
from ..poker.cards import card_str, codes
from ..solvers.holdem_analysis import action_label, spot_state

PREFLOP_HANDS = {
    "AA": ("As", "Ah"), "KK": ("Ks", "Kh"), "TT": ("Ts", "Th"), "66": ("6s", "6h"),
    "22": ("2s", "2h"), "AKs": ("As", "Ks"), "AQo": ("Ad", "Qc"), "AJs": ("Ah", "Jh"),
    "KQs": ("Kd", "Qd"), "T9s": ("Tc", "9c"), "76s": ("7d", "6d"), "A5s": ("Ac", "5c"),
    "K9o": ("Kc", "9h"), "J7o": ("Jc", "7h"), "94o": ("9d", "4s"), "72o": ("7c", "2h"),
}
PREMIUM = ("AA", "KK", "AKs")
TRASH = ("72o", "94o", "J7o")
PREFLOP_SITUATIONS = {          # name -> (position, streets)
    "BTN unopened": ("BTN", ("",)),
    "BB vs limp": ("BB", ("c",)),
    "BB vs raise33": ("BB", ("b33",)),
    "BB vs raise75": ("BB", ("b75",)),
    "BB vs raise150": ("BB", ("b150",)),
    "BB vs all-in": ("BB", ("a",)),
}
FLOPS = {
    "dry_high": ("Kd", "7s", "2h"),
    "wet_connected": ("9h", "8h", "6c"),
    "paired": ("Td", "Tc", "4s"),
    "monotone": ("Qs", "8s", "3s"),
    "low_connected": ("5d", "4c", "3h"),
}
TURN_CARD = {"dry_high": "4d", "wet_connected": "2s", "paired": "Jh", "monotone": "9d",
             "low_connected": "Kh"}
RIVER_CARD = {"dry_high": "Jc", "wet_connected": "Ks", "paired": "6h", "monotone": "2d",
              "low_connected": "8s"}
PREFLOP_LINES = {"limped": "cc", "raise75": "b75c", "raise150": "b150c"}
FLOP_NODES = {                  # name -> (position, current-street actions)
    "BB first": ("BB", ""),
    "BTN after check": ("BTN", "c"),
    "BB facing bet33": ("BB", "cb33"),
    "BB facing bet75": ("BB", "cb75"),
    "BB facing bet150": ("BB", "cb150"),
}
LATER_NODES = {"BB first": ("BB", ""), "BTN after check": ("BTN", "c"),
               "BB facing bet75": ("BB", "cb75")}
CATEGORIES = ("monster", "top_pair_plus", "medium", "weak_made", "draw", "air")
RESERVED = ("2c", "3d")          # left free for the villain filler cards


def category(hole, board) -> Optional[str]:
    f = card_features(hole, board)
    if f.strength >= 5:
        return "monster"
    if f.strength == 4:
        return "top_pair_plus"
    if f.strength == 3:
        return "medium"
    if f.draw >= 2 and f.strength <= 2:
        return "draw"
    if f.strength in (1, 2) and f.draw == 0:
        return "weak_made"
    if f.strength == 0 and f.draw == 0:
        return "air"
    return None


def representative_holes(board: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """First hole (in card-code order) of each category on ``board``."""
    b = codes(board)
    dead = set(b) | set(codes(RESERVED))
    live = [c for c in range(52) if c not in dead]
    out: Dict[str, Tuple[str, str]] = {}
    for h in itertools.combinations(live, 2):
        cat = category(h, b)
        if cat is not None and cat not in out:
            out[cat] = (card_str(h[0]), card_str(h[1]))
        if len(out) == len(CATEGORIES):
            break
    return out


@dataclass(frozen=True)
class MatrixSpot:
    name: str
    street: str
    situation: str
    position: str
    hole: Tuple[str, str]
    board: Tuple[str, ...]
    streets: Tuple[str, ...]
    hand: str                  # preflop class or postflop category
    board_type: str = ""
    preflop_line: str = ""


def canonical_matrix() -> List[MatrixSpot]:
    spots: List[MatrixSpot] = []
    for sit, (pos, streets) in PREFLOP_SITUATIONS.items():
        for hand, hole in PREFLOP_HANDS.items():
            spots.append(MatrixSpot(f"pre|{sit}|{hand}", "preflop", sit, pos, hole, (),
                                    streets, hand))
    for bt, flop in FLOPS.items():
        boards = {"flop": flop, "turn": flop + (TURN_CARD[bt],),
                  "river": flop + (TURN_CARD[bt], RIVER_CARD[bt])}
        for street, board in boards.items():
            reps = representative_holes(board)
            lines = PREFLOP_LINES if street == "flop" else {"raise75": "b75c"}
            nodes = FLOP_NODES if street == "flop" else LATER_NODES
            for lname, pre in lines.items():
                for node, (pos, cur) in nodes.items():
                    prior = {"flop": (pre,), "turn": (pre, "cc"), "river": (pre, "cc", "cc")}[street]
                    for cat in CATEGORIES:
                        if cat not in reps:
                            continue
                        spots.append(MatrixSpot(
                            f"{street}|{bt}|{lname}|{node}|{cat}", street, node, pos,
                            reps[cat], board, prior + (cur,), cat, bt, lname))
    return spots


ACTION_CLASS = {"f": "fold", "c": "passive", "a": "aggressive"}


def action_class(token: str) -> str:
    return ACTION_CLASS.get(token, "aggressive")


Lookup = Callable[[str], Optional[Tuple[Dict[str, float], float]]]


def solver_lookup(solver) -> Lookup:
    def f(key):
        n = solver.infosets.get(key)
        if n is None:
            return None
        v = float(n.strategy_sum.sum())
        if v <= 0:
            return None
        return dict(zip(n.actions, (float(x) for x in n.average_strategy()))), v
    return f


def evaluate_matrix(game, lookup: Lookup, spots: Optional[List[MatrixSpot]] = None) -> List[dict]:
    rows = []
    for sp in spots or canonical_matrix():
        state = spot_state(game, sp.position, sp.hole, sp.board, sp.streets)
        key = game.infoset_key(state)
        legal = game.legal_actions(state)
        hit = lookup(key)
        row = {"name": sp.name, "street": sp.street, "situation": sp.situation,
               "position": sp.position, "hand": sp.hand, "hole": list(sp.hole),
               "board": list(sp.board), "board_type": sp.board_type,
               "preflop_line": sp.preflop_line, "key": key,
               "legal": legal, "labels": [action_label(game, state, a) for a in legal]}
        if hit is None:
            row.update(status="UNVISITED", visits=0.0, policy=None, classes=None)
        else:
            probs, visits = hit
            row.update(status="visited", visits=round(visits, 3),
                       policy={a: round(probs.get(a, 0.0), 4) for a in legal})
            cls = {"fold": 0.0, "passive": 0.0, "aggressive": 0.0}
            for a in legal:
                cls[action_class(a)] += probs.get(a, 0.0)
            row["classes"] = {k: round(v, 4) for k, v in cls.items()}
        rows.append(row)
    return rows
