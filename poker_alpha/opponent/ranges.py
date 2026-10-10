"""Range priors and Bayesian range updates for Hold'em opponents.

Priors come from a versioned, data-driven table
(``poker_alpha/data/preflop_ranges_v1.json`` by default) keyed by stack
bucket, preflop *line* (open / limp / call_open / 3bet / ...) and position.
They are illustrative heuristics; pass your own file to override.

Updates follow Bayes' rule combo-wise::

    P(hand | action) ∝ P(action | hand, state) · P(hand)

with the likelihood supplied by a :class:`~poker_alpha.opponent.behavior.BehaviorModel`
(or a solver-backed :class:`~poker_alpha.opponent.behavior.StrategyLikelihood`).
Ranges are beliefs; :class:`RangeBelief` carries the entropy alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..poker.cards import codes
from ..poker.ranges import (WeightedRange, preflop_strength,
                            strength_percentiles)
from .behavior import BehaviorModel

PRIORS_FORMAT = "pokeralpha.preflop_ranges/v1"


class RangePriors:
    """Preflop range priors loaded from a versioned JSON table."""

    def __init__(self, table: dict) -> None:
        if table.get("format") != PRIORS_FORMAT:
            raise ValueError(f"unsupported priors format {table.get('format')!r}")
        self.table = table
        self._cache: Dict[str, WeightedRange] = {}

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "RangePriors":
        if path is None:
            text = resources.files("poker_alpha.data").joinpath(
                "preflop_ranges_v1.json").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls(json.loads(text))

    def stack_bucket(self, stack_bb: Optional[float]) -> str:
        for b in self.table["stack_buckets"]:
            if b["max_bb"] is None or (stack_bb is not None and stack_bb <= b["max_bb"]):
                return b["name"]
        return self.table["stack_buckets"][-1]["name"]

    def range_text(self, position: str, line: str,
                   stack_bb: Optional[float] = None) -> str:
        pos = self.table.get("position_aliases", {}).get(position, position)
        buckets = [self.stack_bucket(stack_bb), "standard"]
        for bucket in buckets:
            lines = self.table["ranges"].get(bucket, {})
            if line in lines:
                entry = lines[line]
                if pos in entry:
                    return entry[pos]
                if "default" in entry:
                    return entry["default"]
        return "random"

    def prior(self, position: str, line: str,
              stack_bb: Optional[float] = None) -> WeightedRange:
        text = self.range_text(position, line, stack_bb)
        if text not in self._cache:
            self._cache[text] = WeightedRange.from_string(text)
        return self._cache[text]


def classify_preflop_line(actions: Sequence, seat: int) -> str:
    """Name ``seat``'s preflop line from ordered preflop actions.

    ``actions`` are objects with ``seat`` and ``kind`` (fold/check/call/bet/
    raise/all_in/post). Returns one of ``unopened``, ``open``, ``limp``,
    ``call_open``, ``3bet``, ``call_3bet``, ``4bet``, ``call_4bet``,
    ``check_option``. Only the seat's *last voluntary* action is used.
    """
    raises = 0
    line = "unopened"
    for a in actions:
        if a.kind in ("post", "fold") and a.seat != seat:
            continue
        # An "all_in" record is counted as aggressive: a source that only
        # reports "all_in" cannot distinguish an all-in call. Sources that
        # can (the engine, replays) should report such calls as "call".
        aggressive = a.kind in ("bet", "raise", "all_in")
        if a.seat == seat:
            if aggressive:
                line = {0: "open", 1: "3bet"}.get(raises, "4bet")
            elif a.kind == "call":
                line = {0: "limp", 1: "call_open", 2: "call_3bet"}.get(
                    raises, "call_4bet")
            elif a.kind == "check":
                line = "check_option"
        if aggressive:
            raises += 1
    return line


@dataclass(frozen=True)
class RangeBelief:
    """A range plus the uncertainty a consumer must not ignore."""

    range: WeightedRange
    entropy_bits: float
    effective_combos: float
    note: str = ""

    @classmethod
    def of(cls, r: WeightedRange, note: str = "") -> "RangeBelief":
        return cls(r, r.entropy_bits(), r.effective_combos(), note)


def strength_vector(board: Sequence, dead: Iterable = ()) -> np.ndarray:
    """Per-combo strength in [0, 1] for the current street (NaN if blocked)."""
    b = codes(board)
    if not b:
        s = preflop_strength().copy()
        from ..poker.ranges import blocked_mask
        s[blocked_mask(list(codes(dead)))] = np.nan
        return s
    return strength_percentiles(b, dead)


def update_range_for_action(prior: WeightedRange, action: str, *,
                            facing_bet: bool, board: Sequence = (),
                            dead: Iterable = (), model=None,
                            size_pot_fraction: float = 0.5) -> WeightedRange:
    """Posterior range after observing ``action`` (Bayes, combo-wise).

    ``dead`` are cards known not to be in this opponent's hand (hero's cards,
    other revealed cards). Board cards are removed automatically.
    """
    model = model or BehaviorModel()
    b = list(codes(board))
    dead_l = list(codes(dead))
    r = prior.remove_cards(b + dead_l)
    strength = strength_vector(b, dead_l)
    lik = model.likelihood(action, strength, facing_bet, size_pot_fraction)
    return r.update(lik)
