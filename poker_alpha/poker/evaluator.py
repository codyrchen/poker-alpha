"""Texas Hold'em hand evaluator.

Evaluates 5-card hands into a totally ordered strength value and picks the
best 5-card hand out of 5, 6, or 7 cards (hole cards + board).

Two implementations of the same function live here:

* :func:`evaluate_five` — the reference semantics for exactly five cards.
  Straightforward and slow-ish; it defines what a hand value *is*, and the
  test suite pins it against the known 5-card poker frequencies.
* :func:`evaluate_best` — the hot path, used by the Monte Carlo equity
  engine. It evaluates 5-7 cards **directly** from rank/suit histograms
  rather than by taking ``max`` over all C(7,5)=21 five-card subsets.

Profiling (phase 17) showed the exhaustive-subset approach dominated equity
simulation: 94.6% of the runtime, 42 ``evaluate_five`` calls per simulated
hand. The direct evaluator returns bit-identical values — verified
exhaustively against :func:`evaluate_five` over all 2,598,960 five-card
hands, and on large random 6- and 7-card samples — so it is a pure speedup
with no change to any downstream result.

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


def _straight_high_from_ranks(ranks_desc: Sequence[int]) -> int:
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
        high = _straight_high_from_ranks(tuple(ranks))
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


# Precomputed rank/suit for every card code: one list index instead of a
# modulo and a floor-division in the innermost loop of the evaluator.
_RANK_OF = tuple(c % 13 for c in range(52))
_SUIT_OF = tuple(c // 13 for c in range(52))

_ACE_BIT = 1 << 12
_WHEEL_LOW = 0b1111  # deuce..five


def _straight_high_from_mask(rank_mask: int) -> int:
    """High rank of the best straight in a 13-bit rank mask, or -1 if none.

    ``m`` has bit ``j`` set exactly when ranks ``j..j+4`` are all present, so
    the highest set bit of ``m`` is the low card of the best straight.
    """
    m = rank_mask & (rank_mask >> 1) & (rank_mask >> 2) & (rank_mask >> 3) \
        & (rank_mask >> 4)
    if m:
        return m.bit_length() + 3  # (bit_length-1) + 4
    # Wheel: A-2-3-4-5 counts as a straight to the five (high rank 3).
    if rank_mask & _ACE_BIT and rank_mask & _WHEEL_LOW == _WHEEL_LOW:
        return 3
    return -1


def evaluate_best(cards: Iterable) -> HandValue:
    """Best 5-card strength from 5, 6, or 7 cards (codes, Cards, or strings).

    Direct histogram evaluation — equivalent to, but much faster than, taking
    the max over every five-card subset. Categories are tested in descending
    order, so the first match is the best hand.
    """
    cs = codes(cards)
    if len(cs) not in (5, 6, 7):
        raise ValueError(f"need 5-7 cards, got {len(cs)}")
    return evaluate_best_codes(cs)


def evaluate_best_codes(cs: Sequence[int]) -> HandValue:
    """:func:`evaluate_best` for callers that already hold validated codes.

    Skips the mixed-type normalization in :func:`codes`, which profiling
    showed costs ~24% of Monte Carlo equity runtime once the evaluator itself
    is fast — the sampler draws integer codes, so re-checking their types per
    simulation is pure overhead. Callers are responsible for passing 5-7
    in-range, distinct integer codes.
    """
    rank_count = [0] * 13
    suit_count = [0] * 4
    suit_mask = [0, 0, 0, 0]
    rank_mask = 0
    for c in cs:
        r = _RANK_OF[c]
        s = _SUIT_OF[c]
        rank_count[r] += 1
        suit_count[s] += 1
        suit_mask[s] |= 1 << r
        rank_mask |= 1 << r

    # A 7-card hand can hold at most one suit with five or more cards.
    flush_suit = -1
    for s in range(4):
        if suit_count[s] >= 5:
            flush_suit = s
            break

    if flush_suit >= 0:
        high = _straight_high_from_mask(suit_mask[flush_suit])
        if high >= 0:
            return (STRAIGHT_FLUSH, high)

    # Ranks present, highest first, split by multiplicity.
    present = [r for r in range(12, -1, -1) if rank_count[r]]
    quads = [r for r in present if rank_count[r] == 4]
    trips = [r for r in present if rank_count[r] == 3]
    pairs = [r for r in present if rank_count[r] == 2]

    if quads:
        q = quads[0]
        return (FOUR_OF_A_KIND, q, next(r for r in present if r != q))

    if trips:
        t = trips[0]
        # The boat's pair may be a second set of trips or the best pair.
        rest = trips[1:] + pairs
        if rest:
            return (FULL_HOUSE, t, max(rest))

    if flush_suit >= 0:
        fm = suit_mask[flush_suit]
        return (FLUSH, *[r for r in range(12, -1, -1) if fm >> r & 1][:5])

    high = _straight_high_from_mask(rank_mask)
    if high >= 0:
        return (STRAIGHT, high)

    if trips:
        t = trips[0]
        k = [r for r in present if r != t][:2]
        return (THREE_OF_A_KIND, t, k[0], k[1])

    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        return (TWO_PAIR, hi, lo,
                next(r for r in present if r != hi and r != lo))

    if pairs:
        p = pairs[0]
        return (ONE_PAIR, p, *[r for r in present if r != p][:3])

    return (HIGH_CARD, *present[:5])


def compare_hands(hero: Iterable, villain: Iterable) -> int:
    """Compare two (5-7 card) hands: 1 if hero wins, -1 if villain, 0 tie."""
    h, v = evaluate_best(hero), evaluate_best(villain)
    return (h > v) - (h < v)
