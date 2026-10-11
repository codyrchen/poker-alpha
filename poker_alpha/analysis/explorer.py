"""Strategy explorer: query the promoted strategy artifact directly.

Preflop: the classic 13x13 matrix over the 169 canonical hand classes, for
the handful of heads-up preflop nodes the release abstraction actually
represents. Every displayed frequency comes from a live
``SolverStrategyProvider.lookup`` against the loaded artifact — nothing is
hard-coded, and cells where the gate rejects (or the state cannot be
mapped) say so instead of showing numbers.

Postflop: one function that inspects any user-supplied HU state through the
same lookup, labeled ACCEPTED / LOW CONFIDENCE / REJECTED / UNSUPPORTED.

Frequencies shown for a class are computed for one representative combo
(e.g. ``AKs`` -> As Ks); the release encoder buckets preflop hands by rank
pattern and suitedness, so the class is a single information set, but the
representative is still reported for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..decision.solver_gate import ACCEPT, LOW
from ..decision.strategy import LookupMiss, SolverLookup, SolverStrategyProvider
from ..holdem import ManualStateAdapter

RANKS = "AKQJT98765432"

#: Named preflop nodes (HU, 100bb, blinds 0.5/1) that the release tree
#: represents. Each value builds the manual-state dict for given hero cards.
#: Heads-up: the dealer/button posts the small blind and acts first preflop.


def _hu(hero_cards: str, dealer: int, pot: float, actions: list,
        committed: Tuple[float, float], stacks: Tuple[float, float],
        bets: Tuple[float, float]) -> dict:
    return {"num_seats": 2, "hero_seat": 0, "dealer": dealer,
            "small_blind": 0.5, "big_blind": 1.0, "hero_cards": hero_cards,
            "board": "", "pot": pot, "actor": 0,
            "seats": [{"stack": stacks[0], "committed": committed[0],
                       "bet": bets[0]},
                      {"stack": stacks[1], "committed": committed[1],
                       "bet": bets[1]}],
            "actions": actions}


def _btn_first(cards: str) -> dict:
    return _hu(cards, dealer=0, pot=1.5, actions=[],
               committed=(0.5, 1.0), stacks=(99.5, 99.0), bets=(0.5, 1.0))


def _bb_vs_open(cards: str, open_to: float = 2.5) -> dict:
    return _hu(cards, dealer=1, pot=1.0 + open_to,
               actions=[{"street": 0, "seat": 1, "kind": "raise",
                         "amount": open_to}],
               committed=(1.0, open_to), stacks=(99.0, 100.0 - open_to),
               bets=(1.0, open_to))


def _btn_vs_3bet(cards: str, open_to: float = 2.5, three_bet_to: float = 9.0) -> dict:
    return _hu(cards, dealer=0, pot=open_to + three_bet_to,
               actions=[{"street": 0, "seat": 0, "kind": "raise",
                         "amount": open_to},
                        {"street": 0, "seat": 1, "kind": "raise",
                         "amount": three_bet_to}],
               committed=(open_to, three_bet_to),
               stacks=(100.0 - open_to, 100.0 - three_bet_to),
               bets=(open_to, three_bet_to))


PREFLOP_NODES = {
    "BTN first action": _btn_first,
    "BB facing BTN open (to 2.5)": _bb_vs_open,
    "BTN facing BB 3-bet (2.5 → 9)": _btn_vs_3bet,
}


def hand_classes() -> List[List[str]]:
    """The 13x13 grid: diagonal pairs, upper triangle suited, lower offsuit."""
    grid = []
    for i, r1 in enumerate(RANKS):
        row = []
        for j, r2 in enumerate(RANKS):
            if i == j:
                row.append(r1 + r2)
            elif i < j:
                row.append(r1 + r2 + "s")
            else:
                row.append(r2 + r1 + "o")
        grid.append(row)
    return grid


def representative_combo(cls: str) -> str:
    """One concrete combo for a canonical class (``AKs`` -> ``As Ks``)."""
    if len(cls) == 2:                       # pair
        return f"{cls[0]}s {cls[1]}h"
    a, b, suited = cls[0], cls[1], cls[2] == "s"
    return f"{a}s {b}s" if suited else f"{a}s {b}h"


@dataclass(frozen=True)
class CellResult:
    hand: str                              # canonical class, e.g. "AKs"
    combo: str                             # representative combo queried
    status: str                            # ACCEPTED / LOW CONFIDENCE / REJECTED / UNSUPPORTED
    dominant: Optional[str] = None         # highest-frequency action label
    dominant_freq: Optional[float] = None
    mix: Tuple[Tuple[str, float], ...] = ()   # (label, probability), desc
    visits: Optional[float] = None
    reasons: Tuple[str, ...] = ()          # gate / miss reason codes
    notes: Tuple[str, ...] = ()            # e.g. translation note
    signals: Dict[str, Optional[float]] = None  # gate signals (visits,
                                           # movement, seed_disagreement,
                                           # collision) when available


_STATUS = {ACCEPT: "ACCEPTED", LOW: "LOW CONFIDENCE"}


def lookup_cell(solver: SolverStrategyProvider, cls: str,
                node: str = "BTN first action") -> CellResult:
    combo = representative_combo(cls)
    state = ManualStateAdapter.from_dict(PREFLOP_NODES[node](combo))
    res = solver.lookup(state)
    if isinstance(res, LookupMiss):
        status = "REJECTED" if res.code in (
            "LOW_VISIT_COUNT", "HIGH_SEED_DISAGREEMENT", "UNSEEN_STATE",
            "UNSTABLE_ACROSS_CHECKPOINTS", "HIGH_COLLISION_DISPERSION",
            "KNOWN_PATHOLOGICAL_BUCKET") else "UNSUPPORTED"
        return CellResult(cls, combo, status, reasons=tuple(res.reasons),
                          notes=(res.reason,),
                          signals=dict((res.gate or {}).get("signals", {})))
    assert isinstance(res, SolverLookup)
    mix = tuple(sorted(((label, p) for _, label, _, _, p in res.actions),
                       key=lambda x: -x[1]))
    notes = ()
    if not res.exact:
        notes = ("observed sizes were translated onto the trained tree",)
    status = _STATUS.get(res.gate["status"], "REJECTED") if res.gate else "ACCEPTED"
    return CellResult(cls, combo, status,
                      dominant=mix[0][0] if mix else None,
                      dominant_freq=mix[0][1] if mix else None,
                      mix=mix, visits=res.visits,
                      reasons=tuple(res.gate.get("reasons", ())) if res.gate else (),
                      notes=notes,
                      signals=dict((res.gate or {}).get("signals", {})))


def preflop_matrix(solver: SolverStrategyProvider,
                   node: str = "BTN first action") -> List[List[CellResult]]:
    """All 169 cells for a node. Every frequency is a live artifact lookup."""
    return [[lookup_cell(solver, cls, node) for cls in row]
            for row in hand_classes()]


def matrix_stats(matrix: List[List[CellResult]]) -> Dict[str, int]:
    from collections import Counter

    return dict(Counter(c.status for row in matrix for c in row))


# -- postflop ---------------------------------------------------------------

def inspect_state(solver: SolverStrategyProvider, state_dict: dict) -> CellResult:
    """Inspect an arbitrary user-supplied HU state against the artifact.

    The same semantics as the matrix, but for a full state (board, action
    history, stacks). Validation problems surface as UNSUPPORTED with the
    adapter/lookup reason.
    """
    try:
        state = ManualStateAdapter.from_dict(dict(state_dict))
    except (ValueError, KeyError, TypeError) as exc:
        return CellResult("", "", "UNSUPPORTED", notes=(str(exc),))
    from ..holdem import validate

    errors = [i for i in validate(state) if i.severity == "error"]
    if errors:
        return CellResult(state_dict.get("hero_cards", ""), "", "UNSUPPORTED",
                          reasons=tuple(i.code for i in errors),
                          notes=tuple(i.message for i in errors))
    res = solver.lookup(state)
    cards = state_dict.get("hero_cards", "")
    if isinstance(res, LookupMiss):
        status = "UNSUPPORTED" if res.code in (
            "OUTSIDE_ABSTRACTION", "CONFIG_MISMATCH",
            "INCOMPATIBLE_CHECKPOINT") else "REJECTED"
        return CellResult(cards, cards, status, reasons=tuple(res.reasons),
                          notes=(res.reason,))
    mix = tuple(sorted(((label, p) for _, label, _, _, p in res.actions),
                       key=lambda x: -x[1]))
    notes = () if res.exact else (
        "observed sizes were translated onto the trained tree",)
    return CellResult(cards, cards,
                      _STATUS.get(res.gate["status"], "REJECTED")
                      if res.gate else "ACCEPTED",
                      dominant=mix[0][0] if mix else None,
                      dominant_freq=mix[0][1] if mix else None,
                      mix=mix, visits=res.visits,
                      reasons=tuple(res.gate.get("reasons", ())) if res.gate else (),
                      notes=notes)


__all__ = ["RANKS", "PREFLOP_NODES", "CellResult", "hand_classes",
           "representative_combo", "lookup_cell", "preflop_matrix",
           "matrix_stats", "inspect_state"]
