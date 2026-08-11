"""Texas Hold'em hand evaluator.

Evaluates 5-card hands into a totally ordered strength value and picks the
best 5-card hand out of 6 or 7 cards (hole cards + board) by exhaustive
combination — correctness first; the exhaustive step is the documented
optimization target for the later profiling phase.

A hand's strength is a tuple ``(category, tiebreakers...)`` compared
lexicographically, so two hands are compared with plain ``>``/``==``:

===========  =====================  =========================================
category      name                  tiebreakers
===========  =====================  =========================================
8             straight flush        (high rank of the straight,)
7             four of a kind        (quad rank, kicker)
6             full house            (trips rank, pair rank)
5             flush                 5 ranks descending
4             straight              (high rank of the straight,)
3             three of a kind       (trips rank, kicker, kicker)
2             two pair              (high pair, low pair, kicker)
1             one pair              (pair rank, kicker, kicker, kicker)
0             high card             5 ranks descending
===========  =====================  =========================================

Ranks are 0 (deuce) .. 12 (ace). The wheel (A-2-3-4-5) counts as a straight
with high rank 3 (the five), the lowest possible straight.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Tuple

from .cards import codes

HandValue = Tuple[int, ...]

STRAIGHT_FLUSH = 8
FOUR_OF_A_KIND = 7
FULL_HOUSE = 6
FLUSH = 5
STRAIGHT = 4
THREE_OF_A_KIND = 3
TWO_PAIR = 2
ONE_PAIR = 1
HIGH_CARD = 0

CATEGORY_NAMES = {
    8: "straight flush",
    7: "four of a kind",
    6: "full house",
    5: "flush",
    4: "straight",
    3: "three of a kind",
    2: "two pair",
    1: "one pair",
    0: "high card",
}


def _straight_high(ranks_desc: Sequence[int]) -> int:
    """High card of the straight formed by 5 distinct ranks, or -1 if none."""
    if ranks_desc[0] - ranks_desc[4] == 4:
        return ranks_desc[0]
    # Wheel: A(12) 5(3) 4(2) 3(1) 2(0) -> straight to the five (high rank 3).
    if ranks_desc == (12, 3, 2, 1, 0):
        return 3
    return -1


def evaluate_five(cards: Sequence[int]) -> HandValue:
    """Strength of exactly five cards (integer codes). Higher tuple wins."""
    ranks = sorted((c % 13 for c in cards), reverse=True)
    suits = {c // 13 for c in cards}
    is_flush = len(suits) == 1

    # Count multiplicities: pairs/trips/quads.
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Sort by (count, rank) descending: primary groups first.
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    distinct = len(groups)
    if distinct == 5:
        high = _straight_high(tuple(ranks))
        if high >= 0:
            return (STRAIGHT_FLUSH if is_flush else STRAIGHT, high)
        if is_flush:
            return (FLUSH, *ranks)
        return (HIGH_CARD, *ranks)

    if distinct == 2:
        (r1, n1), (r2, _) = groups
        if n1 == 4:
            return (FOUR_OF_A_KIND, r1, r2)
        return (FULL_HOUSE, r1, r2)  # 3 + 2

    if distinct == 3:
        (r1, n1), (r2, _), (r3, _) = groups
        if n1 == 3:
            return (THREE_OF_A_KIND, r1, max(r2, r3), min(r2, r3))
        return (TWO_PAIR, r1, r2, r3)  # sorted: two pairs then kicker

    # distinct == 4: one pair.
    (r1, _), (r2, _), (r3, _), (r4, _) = groups
    return (ONE_PAIR, r1, r2, r3, r4)


def evaluate_best(cards: Iterable) -> HandValue:
    """Best 5-card strength from 5, 6, or 7 cards (codes, Cards, or strings)."""
    cs = codes(cards)
    if len(cs) == 5:
        return evaluate_five(cs)
    if len(cs) not in (6, 7):
        raise ValueError(f"need 5-7 cards, got {len(cs)}")
    return max(evaluate_five(combo) for combo in combinations(cs, 5))


def compare_hands(hero: Iterable, villain: Iterable) -> int:
    """Compare two (5-7 card) hands: 1 if hero wins, -1 if villain, 0 tie."""
    h, v = evaluate_best(hero), evaluate_best(villain)
    return (h > v) - (h < v)
