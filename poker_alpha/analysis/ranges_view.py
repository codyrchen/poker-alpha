"""Opponent-range visualization models (pure Python, no UI).

Exposes the EXISTING weighted-range machinery as 13x13 matrices and
street-by-street stages. These are **MODELLED OPPONENT RANGES** — Bayesian
updates of a position/line prior under a chosen behavior archetype — and
are never ground truth about an opponent's actual holding. Every surface
rendering them is required to say so.

``range_stages`` reproduces exactly what the decision engine computes
(:func:`poker_alpha.decision.recommend.infer_opponent_range`), just
truncated at each street boundary so the evolution
``prior → flop update → turn update → river update`` is visible. Tests
pin the displayed matrices to the underlying :class:`WeightedRange`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from ..decision.recommend import BOARD_AT_STREET, infer_opponent_range
from ..holdem.observed import ObservedTableState
from ..holdem.state import Street
from ..opponent.behavior import ARCHETYPE_MODELS
from ..opponent.ranges import RangePriors
from ..poker.ranges import WeightedRange
from .explorer import RANKS, hand_classes

STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


def range_matrix(wr: WeightedRange) -> List[List[float]]:
    """13x13 class-weight matrix in the standard grid order (pairs on the
    diagonal, suited upper, offsuit lower). Cells sum to ~1 for a
    normalized range. Pinned to ``WeightedRange.class_distribution``."""
    dist = wr.class_distribution()
    return [[dist.get(cls, 0.0) for cls in row] for row in hand_classes()]


@dataclass(frozen=True)
class RangeStage:
    name: str                      # "Prior (BTN, open)" / "After Flop" ...
    street: int                    # -1 for the prior
    range: WeightedRange
    matrix: List[List[float]]
    entropy_bits: float
    effective_combos: float
    live_combos: int
    warnings: Tuple[str, ...]


def _stage(name: str, street: int, wr: WeightedRange,
           warnings: List[str]) -> RangeStage:
    return RangeStage(name=name, street=street, range=wr,
                      matrix=range_matrix(wr),
                      entropy_bits=wr.entropy_bits(),
                      effective_combos=wr.effective_combos(),
                      live_combos=wr.num_live_combos,
                      warnings=tuple(warnings))


def range_stages(state: ObservedTableState, seat: int,
                 model: str = "regular",
                 priors: Optional[RangePriors] = None) -> List[RangeStage]:
    """The modelled range for ``seat`` after each street's actions.

    Stage 0 is the preflop prior+line (post card removal); stage k is the
    posterior after every action up to and including street k, computed by
    the same engine function the decision pipeline uses, on the state
    truncated to that street (board sliced to match, so postflop updates
    see only the cards that were out at the time).
    """
    pri = priors or RangePriors.load()
    bm = ARCHETYPE_MODELS[model]
    streets = sorted({a.street for a in state.action_history}) or [0]
    stages: List[RangeStage] = []
    # prior: position/line with no postflop updates — street-0-only history
    pre_hist = tuple(a for a in state.action_history if a.street == 0)
    w0: List[str] = []
    pre_state = replace(state, action_history=pre_hist,
                        board=(), street=Street.PREFLOP)
    r0, line = infer_opponent_range(pre_state, seat, pri, bm, w0)
    stages.append(_stage(f"Preflop ({line})", 0, r0, w0))
    for s in [s for s in streets if s >= 1]:
        hist = tuple(a for a in state.action_history if a.street <= s)
        w: List[str] = []
        st = replace(state, action_history=hist,
                     board=state.board[:BOARD_AT_STREET[min(s, 3)]],
                     street=Street(min(s, 3)))
        r, _ = infer_opponent_range(st, seat, pri, bm, w)
        stages.append(_stage(f"After {STREET_NAMES.get(s, s)}", s, r, w))
    return stages


def matrix_cells(matrix: List[List[float]]) -> List[Tuple[str, float]]:
    """Flat (class, weight) pairs in grid order, for tabular displays."""
    grid = hand_classes()
    return [(grid[i][j], matrix[i][j])
            for i in range(13) for j in range(13)]


__all__ = ["range_matrix", "RangeStage", "range_stages", "matrix_cells",
           "STREET_NAMES", "RANKS"]
