"""Canonical heads-up spots queried repeatedly across training checkpoints.

Each spot is a fixed (position, hole cards, board, action history) in the
abstract game; :func:`spot_policies` reads the solver's average strategy at
the corresponding information set together with its visit count, so policy
movement can be tracked across checkpoints and compared across seeds.
Unvisited information sets are reported as such (regret matching's default
is uniform) rather than as a learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..games.holdem import HoldemGame
from ..solvers.holdem_analysis import action_label, spot_state

FLOP = ("Qs", "7s", "2d")


@dataclass(frozen=True)
class Spot:
    name: str
    group: str
    position: str
    hole: Tuple[str, str]
    board: Tuple[str, ...] = ()
    streets: Tuple[str, ...] = ("",)


def canonical_spots() -> List[Spot]:
    pre = [Spot(f"BTN open {n}", "preflop unopened", "BTN", h)
           for n, h in (("AA", ("As", "Ah")), ("AKs", ("As", "Ks")),
                        ("AKo", ("Ad", "Kc")), ("72o", ("7c", "2h")))]
    # BTN raises 75% pot, BB calls; flop Qs 7s 2d. BB acts first.
    bb_first = [Spot(f"BB flop first {n}", "BB first to act on flop", "BB", h, FLOP,
                     ("b75c", ""))
                for n, h in (("top pair", ("Qh", "Jd")),
                             ("nut flush draw", ("As", "5s")),
                             ("air", ("9h", "8c")))]
    btn_after_check = [Spot(f"BTN flop after check {n}", "BTN vs BB check", "BTN", h,
                            FLOP, ("b75c", "c"))
                       for n, h in (("top pair", ("Qh", "Jd")),
                                    ("nut flush draw", ("As", "5s")),
                                    ("air", ("9h", "8c")))]
    # BB checks, BTN bets 75%, BB faces the bet.
    facing = [Spot(f"BB facing c-bet {n}", "BB facing flop bet", "BB", h, FLOP,
                   ("b75c", "cb75"))
              for n, h in (("strong made (set)", ("7h", "7d")),
                           ("medium (middle pair)", ("7c", "6h")),
                           ("draw (nut flush draw)", ("As", "5s")),
                           ("air", ("9h", "8c")))]
    return pre + bb_first + btn_after_check + facing


@dataclass(frozen=True)
class SpotPolicy:
    spot: str
    key: str
    visits: float
    visited: bool
    actions: Tuple[str, ...]          # human labels
    probs: Tuple[float, ...]

    @property
    def l1_from_uniform(self) -> float:
        p = np.array(self.probs)
        return float(np.abs(p - 1.0 / len(p)).sum())


def spot_policies(game: HoldemGame, infosets: Dict[str, object],
                  spots: Sequence[Spot]) -> List[SpotPolicy]:
    out = []
    for sp in spots:
        state = spot_state(game, sp.position, sp.hole, sp.board, sp.streets)
        key = game.infoset_key(state)
        acts = game.legal_actions(state)
        labels = tuple(action_label(game, state, a) for a in acts)
        node = infosets.get(key)
        if node is None:
            probs = tuple([1.0 / len(acts)] * len(acts))
            visits = 0.0
        else:
            probs = tuple(float(x) for x in node.average_strategy())
            visits = float(node.strategy_sum.sum())
        out.append(SpotPolicy(sp.name, key, visits, node is not None and visits > 0,
                              labels, probs))
    return out


def l1(a: SpotPolicy, b: SpotPolicy) -> Optional[float]:
    if a.actions != b.actions:
        return None
    return float(np.abs(np.array(a.probs) - np.array(b.probs)).sum())
