"""Cheap, deterministic card features for solver hot paths (Phase 26A).

Unlike :func:`~poker_alpha.abstraction.cards.hand_features` (which runs
Monte Carlo equity and enumerates every opponent combo), everything here
costs a handful of hand evaluations, is exact and deterministic, and is
cached per ``(hole, board)``. It is meant to be called at every MCCFR
infoset lookup.

Reference semantics (``CARD_FEATURES_VERSION`` = 1)
--------------------------------------------------
Features describe the *made hand relative to the board* and the *draws*,
not equity against any range. No reference opponent range is involved; the
only reference is the board itself (top pair = pairs the highest board
rank, etc.). Offline evaluation measures how well these classes separate
equity vs a uniformly random hand (``hand_equity``), which is the explicit
reference range for quality metrics only.

Strength ladder (``strength``, 0..7)
-----------------------------------
Ordered by mean equity vs a random hand on a measured river corpus
(Phase 26; ``docs/abstraction.md``):

==  =====================================================================
0   air (jack high or worse), board plays
1   king/queen high, ace high, two overcards, underpair, board trips
2   bottom pair, pocket pair between board ranks
3   middle pair, top pair with a weak kicker
4   top pair good kicker (J+), overpair, two pair with both hole cards
5   set, trips, straight
6   flush, full house
7   quads, straight flush
==  =====================================================================

``nut`` is 2 for hands that cannot currently be beaten by any holding
(checked exactly but cheaply: see :func:`_is_current_nuts`), 1 for
strength >= 5 otherwise, 0 below.

``draw``: 0 none, 1 weak (gutshot or backdoor flush), 2 strong (open-ended
or flush draw), 3 combo (flush draw + straight draw). Only before the
river and only for hands below a straight.

``blocker``: 0 none, 1 holds a card of a 3+-suited board's suit, 2 holds
the highest missing card of that suit (nut-flush blocker).

``potential``: 0 none (river, or no draw), 1 weak draw, 2 strong/combo draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple

from ..poker.evaluator import evaluate_best_codes
from .cards import board_texture, draw_type, preflop_class, texture_code

CARD_FEATURES_VERSION = 1

MADE_STRENGTH = {
    "air": 0, "board_plays": 0,
    "air_high": 1, "overcards": 1, "ace_high": 1, "underpair": 1,
    "board_trips": 1,
    "bottom_pair": 2, "pocket_middle": 2,
    "middle_pair": 3, "top_pair_weak": 3,
    "top_pair_good": 4, "overpair": 4, "two_pair": 4,
    "set": 5, "trips": 5, "straight": 5,
    "flush": 6, "full_house": 6,
    "quads": 7, "straight_flush": 7,
}
DRAW_CLASS = {"none": 0, "backdoor_flush": 1, "gutshot": 1, "open_ended": 2,
              "flush_draw": 2, "combo_draw": 3}


@dataclass(frozen=True)
class CardFeatures:
    street: int            # 0 preflop .. 3 river
    made: str              # made-hand class name (postflop) or preflop class
    strength: int          # 0..7 ladder (preflop: tier 0..7 from class equity)
    draw: int              # 0..3
    nut: int               # 0..2
    blocker: int           # 0..2
    texture: str           # texture_code v1 ("pre" preflop)
    potential: int         # 0..2


def _rank_counts(cards: Sequence[int]):
    counts = {}
    for c in cards:
        counts[c % 13] = counts.get(c % 13, 0) + 1
    return counts


def made_hand_class(hole: Sequence[int], board: Sequence[int]) -> str:
    """Made-hand class relative to the board (see module docstring)."""
    h, b = list(hole), list(board)
    val = evaluate_best_codes(h + b)
    cat = val[0]
    if len(b) == 5 and val == evaluate_best_codes(b):
        return "board_plays"
    hr = sorted((c % 13 for c in h), reverse=True)
    bcounts = _rank_counts(b)
    branks = sorted(bcounts, reverse=True)
    top = branks[0]
    if cat == 8:
        return "straight_flush"
    if cat == 7:
        # Quads on board with no hole contribution is board strength.
        if max(bcounts.values()) == 4 and not any(r in bcounts for r in hr):
            return "board_plays"
        return "quads"
    if cat == 6:
        return "full_house"
    if cat == 5:
        return "flush"
    if cat == 4:
        return "straight"
    if cat == 3:
        if hr[0] == hr[1] and bcounts.get(hr[0], 0) == 1:
            return "set"
        if max(bcounts.values()) == 3 and not any(r in bcounts for r in hr):
            return "board_trips"
        return "trips"
    paired_with_board = [r for r in set(hr) if r in bcounts]
    pocket = hr[0] == hr[1]
    if cat == 2:
        if not pocket and len(paired_with_board) == 2:
            return "two_pair"
        # One real pair plus a board pair: classify by hero's own pair.
    if pocket:
        if hr[0] > top:
            return "overpair"
        if hr[0] < branks[-1]:
            return "underpair"
        return "pocket_middle"
    if paired_with_board:
        r = max(paired_with_board)
        if r == top:
            kicker = hr[1] if hr[0] == r else hr[0]
            return "top_pair_good" if kicker >= 9 else "top_pair_weak"
        if len(branks) > 1 and r == branks[1]:
            return "middle_pair"
        return "bottom_pair"
    if hr[1] > top:
        return "overcards"
    if hr[0] == 12:
        return "ace_high"
    if hr[0] >= 10:                 # king or queen high
        return "air_high"
    return "air"


@lru_cache(maxsize=None)
def _preflop_tier(cls: str) -> int:
    """0..7 tier of a 169 class by its equity vs a random hand."""
    from ..poker.ranges import CLASS_MEMBERS, preflop_strength

    s = float(preflop_strength()[CLASS_MEMBERS[cls][0]])
    return min(int(s * 8), 7)


def _is_current_nuts(hole: Sequence[int], board: Sequence[int]) -> bool:
    """True if no two-card holding beats hero right now.

    Exact, but avoids enumerating 1,000+ combos: only holdings that could
    possibly reach hero's category matter. Callers only ask for strength
    >= 5 hands (sets and better), keeping the ~1,000-evaluation scan rare;
    the result is cached with the rest of the features.
    """
    from itertools import combinations

    h, b = list(hole), list(board)
    hv = evaluate_best_codes(h + b)
    dead = set(h) | set(b)
    live = [c for c in range(52) if c not in dead]
    for a, c in combinations(live, 2):
        if evaluate_best_codes([a, c] + b) > hv:
            return False
    return True


def _flush_blockers(hole: Sequence[int], board: Sequence[int]) -> Tuple[bool, bool]:
    """``(holds a card of the board's most frequent suit, holds the highest
    missing card of it)`` for boards with 2+ cards of one suit — the flush
    part of :func:`.cards._blockers`, without its straight-blocker scan."""
    suit_counts = [0] * 4
    for c in board:
        suit_counts[c // 13] += 1
    fb = nfb = False
    ms = max(suit_counts)
    if ms >= 2 and len(board) >= 3:
        for s in range(4):
            if suit_counts[s] != ms:
                continue
            mine = {c % 13 for c in hole if c // 13 == s}
            if mine:
                fb = True
                on_board = {c % 13 for c in board if c // 13 == s}
                if max(r for r in range(13) if r not in on_board) in mine:
                    nfb = True
    return fb, nfb


@lru_cache(maxsize=1_000_000)
def _features(hole: Tuple[int, int], board: Tuple[int, ...]) -> CardFeatures:
    if not board:
        cls = preflop_class(hole)
        return CardFeatures(0, cls, _preflop_tier(cls), 0, 0, 0, "pre", 0)
    street = {3: 1, 4: 2, 5: 3}[len(board)]
    made = made_hand_class(hole, board)
    strength = MADE_STRENGTH[made]
    draw = DRAW_CLASS[draw_type(hole, board)] if street < 3 and strength < 5 else 0
    if strength >= 5:
        nut = 2 if _is_current_nuts(hole, board) else 1
    else:
        nut = 0
    fb, nfb = _flush_blockers(hole, board)
    blocker = 2 if nfb else (1 if fb else 0)
    potential = 0 if street == 3 else (0 if draw == 0 else (1 if draw == 1 else 2))
    tex = texture_code(board_texture(board), 1)
    return CardFeatures(street, made, strength, draw, nut, blocker, tex, potential)


def card_features(hole: Sequence[int], board: Sequence[int] = ()) -> CardFeatures:
    """Cached :class:`CardFeatures` (order of hole/board cards irrelevant
    except flop vs turn vs river grouping, which only depends on length)."""
    return _features(tuple(sorted(hole)), tuple(sorted(board)))


TRANSITION_VERSION = 1


def transition_label(prev: CardFeatures, cur: CardFeatures) -> str:
    """Strategic transition between consecutive streets (turn, river).

    ``I``/``S``/``W`` improved / same / weakened strength class, then a draw
    event ``g`` gained, ``c`` completed (had a draw and reached strength >= 5),
    ``m`` missed (had a draw, river, not completed), ``k`` kept, ``-`` none;
    then ``+`` became current nuts, ``x`` lost the nuts, ``=`` otherwise; then
    ``T`` if the board gained a flush or paired (major texture shift) else
    ``.``.
    """
    s = "I" if cur.strength > prev.strength else ("W" if cur.strength < prev.strength else "S")
    if prev.draw and cur.strength >= 5 and cur.strength > prev.strength:
        d = "c"
    elif prev.draw and cur.street == 3:
        d = "m"
    elif prev.draw and cur.draw:
        d = "k"
    elif not prev.draw and cur.draw:
        d = "g"
    else:
        d = "-"
    n = "+" if cur.nut == 2 and prev.nut < 2 else ("x" if prev.nut == 2 and cur.nut < 2 else "=")
    shift = (cur.texture[0] == "p" and prev.texture[0] != "p") or \
        (cur.texture[1] == "f" and prev.texture[1] != "f")
    return f"{s}{d}{n}{'T' if shift else '.'}"
