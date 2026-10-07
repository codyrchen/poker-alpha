"""Interpretable Hold'em card abstraction.

Three layers, deliberately kept separate:

1. **Preflop classes** — the 169 strategically distinct starting hands
   (13 pairs, 78 suited, 78 offsuit). Preflop, suits are fully symmetric, so
   this map is *lossless* up to suit isomorphism.
2. **Features** — :class:`BoardTexture` and :class:`HandFeatures` describe a
   hand/board in human terms (made hand, draws, nut potential, blockers,
   texture, equity, SPR, position). They are plain typed values: nothing is
   collapsed into an opaque string at this layer.
3. **Buckets** — :class:`HoldemBucketEncoder` (in :mod:`.holdem`) chooses a
   small, configurable subset of features and discretizes them.

Determinism: equity estimates are Monte Carlo but seeded from a stable hash
of the *suit-canonical* cards, so suit-isomorphic inputs give bit-identical
features and every call is reproducible.

Nothing here claims to be an optimal abstraction; it is a defensible,
inspectable starting point.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..poker.cards import RANK_CHARS, codes
from ..poker.evaluator import evaluate_best_codes

# -- preflop classes -------------------------------------------------------


def preflop_class(cards: Sequence) -> str:
    """Canonical starting-hand class, e.g. ``"AA"``, ``"AKs"``, ``"72o"``."""
    a, b = codes(cards)
    if a == b:
        raise ValueError("duplicate hole cards")
    ra, rb = a % 13, b % 13
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANK_CHARS[hi] * 2
    suited = a // 13 == b // 13
    return f"{RANK_CHARS[hi]}{RANK_CHARS[lo]}{'s' if suited else 'o'}"


def all_preflop_classes() -> List[str]:
    """All 169 classes, strongest-rank-first (pairs, then suited/offsuit)."""
    out: List[str] = []
    for hi in range(12, -1, -1):
        out.append(RANK_CHARS[hi] * 2)
    for hi in range(12, -1, -1):
        for lo in range(hi - 1, -1, -1):
            out.append(f"{RANK_CHARS[hi]}{RANK_CHARS[lo]}s")
            out.append(f"{RANK_CHARS[hi]}{RANK_CHARS[lo]}o")
    return out


def combos_for_class(cls: str) -> List[Tuple[int, int]]:
    """All concrete ``(low_code, high_code)`` combos in a preflop class."""
    r1 = RANK_CHARS.index(cls[0])
    r2 = RANK_CHARS.index(cls[1])
    out = []
    for s1 in range(4):
        for s2 in range(4):
            a, b = s1 * 13 + r1, s2 * 13 + r2
            if a == b:
                continue
            if len(cls) == 2:          # pair
                if s1 >= s2:
                    continue
            elif cls[2] == "s":
                if s1 != s2:
                    continue
            else:
                if s1 == s2:
                    continue
            out.append((min(a, b), max(a, b)))
    return sorted(set(out))


PREFLOP_CLASSES: Tuple[str, ...] = tuple(all_preflop_classes())
PREFLOP_CLASS_INDEX: Dict[str, int] = {c: i for i, c in enumerate(PREFLOP_CLASSES)}

# -- suit canonicalization -------------------------------------------------

_SUIT_PERMS = tuple(itertools.permutations(range(4)))


def _permute(cards: Sequence[int], perm: Sequence[int]) -> Tuple[int, ...]:
    return tuple(perm[c // 13] * 13 + c % 13 for c in cards)


def canonicalize_suits(hole: Sequence[int],
                       streets: Sequence[Sequence[int]] = ()) -> Tuple[Tuple[int, ...], ...]:
    """Suit-isomorphism canonical form of hole cards + board streets.

    ``streets`` groups board cards by deal (flop, turn, river); order within a
    group is irrelevant, order between groups is preserved. Returns
    ``(sorted hole, sorted flop, turn, river...)`` under the suit relabelling
    that makes this tuple lexicographically smallest.
    """
    groups = [tuple(hole)] + [tuple(g) for g in streets]
    best = None
    for perm in _SUIT_PERMS:
        cand = tuple(tuple(sorted(_permute(g, perm))) for g in groups)
        if best is None or cand < best:
            best = cand
    return best


def board_streets(board: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Split a 0-5 card board into its (flop, turn, river) deal groups."""
    board = tuple(board)
    if not board:
        return ()
    groups = [board[:3]]
    groups.extend((c,) for c in board[3:])
    return tuple(groups)


def stable_seed(*parts: Tuple[int, ...], salt: str = "") -> int:
    """A platform-stable 64-bit seed from integer tuples (no ``hash()``)."""
    h = hashlib.sha256(salt.encode("utf-8"))
    for part in parts:
        h.update(bytes([len(part)]) + bytes(int(x) for x in part))
    return int.from_bytes(h.digest()[:8], "little")


# -- board texture ---------------------------------------------------------

PAIREDNESS = ("unpaired", "paired", "two_pair", "trips", "full_house", "quads")
SUITEDNESS = ("none", "rainbow", "two_tone", "monotone", "three_flush",
              "four_flush", "five_flush")
HIGH_CARD = ("low", "middle", "broadway", "ace")


@dataclass(frozen=True)
class BoardTexture:
    """Human-readable description of the community cards."""

    num_cards: int
    pairedness: str
    suitedness: str
    max_suit_count: int
    connectedness: int       # max distinct board ranks inside any 5-rank window
    straight_possible: bool  # some two hole cards make a straight
    flush_possible: bool     # some hole cards make a flush
    high_card: str           # low (<=8) / middle (9-J) / broadway (Q-K) / ace


def board_texture(board: Sequence) -> BoardTexture:
    b = codes(board)
    if len(b) > 5 or len(set(b)) != len(b):
        raise ValueError("board must be 0-5 distinct cards")
    if not b:
        return BoardTexture(0, "unpaired", "none", 0, 0, False, False, "low")
    ranks = [c % 13 for c in b]
    counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    if counts[0] == 4:
        paired = "quads"
    elif counts[0] == 3:
        paired = "full_house" if len(counts) > 1 and counts[1] >= 2 else "trips"
    elif counts[0] == 2:
        paired = "two_pair" if len(counts) > 1 and counts[1] == 2 else "paired"
    else:
        paired = "unpaired"
    suit_counts = [0, 0, 0, 0]
    for c in b:
        suit_counts[c // 13] += 1
    ms = max(suit_counts)
    if len(b) < 3:
        suited = "none"
    elif ms >= 5:
        suited = "five_flush"
    elif ms == 4:
        suited = "four_flush"
    elif ms == 3:
        suited = "monotone" if len(b) == 3 else "three_flush"
    elif ms == 2:
        suited = "two_tone"
    else:
        suited = "rainbow"
    # Window over ranks with the ace also playing low (index -1 -> 0).
    rset = set(ranks)
    ext = {r + 1 for r in rset} | ({0} if 12 in rset else set())
    conn = max(len([r for r in ext if lo <= r < lo + 5]) for lo in range(0, 10))
    straight_possible = conn >= 3 and len(b) >= 3
    flush_possible = ms >= 3
    top = max(ranks)
    if top == 12:
        high = "ace"
    elif top >= 10:
        high = "broadway"
    elif top >= 7:
        high = "middle"
    else:
        high = "low"
    return BoardTexture(len(b), paired, suited, ms, conn, straight_possible,
                        flush_possible, high)


def texture_code(t: BoardTexture, version: int = 1) -> str:
    """Compact discrete texture label used by bucket encoders.

    Version 1: ``<pair class><suit class><connected>`` where pair class is
    ``u``/``p`` (unpaired / any pairing), suit class ``r``/``t``/``f``
    (rainbow / flush draw possible / flush possible) and connected ``c``/``d``
    (straight possible or not).
    """
    if version != 1:
        raise ValueError(f"unknown texture version {version}")
    if t.num_cards == 0:
        return "pre"
    p = "u" if t.pairedness == "unpaired" else "p"
    if t.max_suit_count >= 3:
        s = "f"
    elif t.max_suit_count == 2 and t.num_cards < 5:
        s = "t"
    else:
        s = "r"
    c = "c" if t.straight_possible else "d"
    return p + s + c


# -- hand features ---------------------------------------------------------

DRAWS = ("none", "backdoor_flush", "gutshot", "open_ended", "flush_draw",
         "combo_draw")


def _has_straight(ranks) -> bool:
    mask = 0
    for r in ranks:
        mask |= 1 << r
    full = (mask << 1) | (1 if mask & (1 << 12) else 0)
    return (full & (full >> 1) & (full >> 2) & (full >> 3) & (full >> 4)) != 0


def straight_completing_ranks(hole: Sequence[int], board: Sequence[int]) -> List[int]:
    """Ranks that would give hero a straight that uses a hole card."""
    hr = {c % 13 for c in hole}
    br = {c % 13 for c in board}
    if _has_straight(hr | br):
        return []
    out = []
    for x in range(13):
        if _has_straight(hr | br | {x}) and not _has_straight(br | {x}):
            out.append(x)
    return out


def draw_type(hole: Sequence[int], board: Sequence[int]) -> str:
    """Strongest draw hero holds (only meaningful before the river)."""
    if len(board) < 3 or len(board) >= 5:
        return "none"
    cards = list(hole) + list(board)
    if evaluate_best_codes(cards)[0] >= 4:  # already straight or better
        return "none"
    suit_total = [0] * 4
    suit_hole = [0] * 4
    for c in cards:
        suit_total[c // 13] += 1
    for c in hole:
        suit_hole[c // 13] += 1
    flush_draw = any(suit_total[s] == 4 and suit_hole[s] for s in range(4))
    backdoor = len(board) == 3 and any(
        suit_total[s] == 3 and suit_hole[s] for s in range(4))
    outs = straight_completing_ranks(hole, board)
    straight = "open_ended" if len(outs) >= 2 else ("gutshot" if outs else "")
    if flush_draw and straight:
        return "combo_draw"
    if flush_draw:
        return "flush_draw"
    if straight:
        return straight
    if backdoor:
        return "backdoor_flush"
    return "none"


@lru_cache(maxsize=200_000)
def _strength_vs_all(hole: Tuple[int, ...], board: Tuple[int, ...]) -> Tuple[float, int, int]:
    """Exact current strength vs every opponent combo on this board.

    Returns ``(fraction beaten + half ties, combos ahead of hero, total)``.
    """
    dead = set(hole) | set(board)
    live = [c for c in range(52) if c not in dead]
    hv = evaluate_best_codes(list(hole) + list(board))
    win = tie = ahead = total = 0
    bl = list(board)
    for a, b in itertools.combinations(live, 2):
        ov = evaluate_best_codes([a, b] + bl)
        total += 1
        if hv > ov:
            win += 1
        elif hv == ov:
            tie += 1
        else:
            ahead += 1
    return (win + 0.5 * tie) / total, ahead, total


@lru_cache(maxsize=200_000)
def _canonical_equity(canon: Tuple[Tuple[int, ...], ...], samples: int) -> float:
    """Seeded MC equity vs a uniformly random hand for canonical cards."""
    hole = canon[0]
    board = tuple(c for g in canon[1:] for c in g)
    rng = np.random.default_rng(stable_seed(*canon, salt=f"eq{samples}"))
    dead = set(hole) | set(board)
    live = np.array([c for c in range(52) if c not in dead], dtype=np.int64)
    need = 5 - len(board)
    hl, bl = list(hole), list(board)
    score = 0.0
    for _ in range(samples):
        pick = live[rng.choice(len(live), size=2 + need, replace=False)]
        opp = [int(pick[0]), int(pick[1])]
        run = bl + [int(x) for x in pick[2:]]
        hv = evaluate_best_codes(hl + run)
        ov = evaluate_best_codes(opp + run)
        score += 1.0 if hv > ov else (0.5 if hv == ov else 0.0)
    return score / samples


def hand_equity(hole: Sequence, board: Sequence = (), samples: int = 400) -> float:
    """Deterministic, suit-invariant equity vs a random hand.

    On the river this is exact (enumerated); otherwise a seeded Monte Carlo
    estimate with ``samples`` runouts (std. error <= 0.5/sqrt(samples)).
    """
    h, b = codes(hole), codes(board)
    if len(b) == 5:
        return _strength_vs_all(tuple(sorted(h)), tuple(sorted(b)))[0]
    canon = canonicalize_suits(h, board_streets(b))
    return _canonical_equity(canon, int(samples))


def _blockers(hole: Sequence[int], board: Sequence[int]) -> Tuple[bool, bool, float]:
    """``(holds flush-suit card, holds nut-flush card, straight blocker share)``."""
    suit_counts = [0] * 4
    for c in board:
        suit_counts[c // 13] += 1
    flush_blocker = nut_flush_blocker = False
    ms = max(suit_counts) if board else 0
    if ms >= 2 and len(board) >= 3:
        for s in range(4):
            if suit_counts[s] != ms:
                continue
            on_board = {c % 13 for c in board if c // 13 == s}
            mine = {c % 13 for c in hole if c // 13 == s}
            if mine:
                flush_blocker = True
                top_missing = max(r for r in range(13) if r not in on_board)
                if top_missing in mine:
                    nut_flush_blocker = True
    # Straight blockers: share of two-rank opponent holdings that make a
    # straight with the board which use at least one of hero's ranks.
    br = {c % 13 for c in board}
    hr = {c % 13 for c in hole}
    pairs = blocked = 0
    if len(board) >= 3:
        for a in range(13):
            for b in range(a, 13):
                if _has_straight(br | {a, b}) and not _has_straight(br):
                    pairs += 1
                    if a in hr or b in hr:
                        blocked += 1
    share = blocked / pairs if pairs else 0.0
    return flush_blocker, nut_flush_blocker, share


@dataclass(frozen=True)
class HandFeatures:
    """Interpretable description of one player's holding in context."""

    street: int                  # 0 preflop .. 3 river
    preflop_class: str
    hand_category: int           # evaluator category (0 high card .. 8 SF); -1 preflop
    equity: float                # vs a uniformly random hand
    current_strength: float      # exact share of opponent combos beaten now
    combos_ahead: int            # opponent combos currently beating hero
    is_nuts: bool                # no opponent combo currently beats hero
    draw: str
    flush_blocker: bool
    nut_flush_blocker: bool
    straight_blocker_share: float
    texture: BoardTexture
    spr: Optional[float] = None
    position: Optional[str] = None
    effective_stack: Optional[float] = None


def hand_features(hole: Sequence, board: Sequence = (), *,
                  pot: Optional[float] = None,
                  effective_stack: Optional[float] = None,
                  position: Optional[str] = None,
                  equity_samples: int = 400) -> HandFeatures:
    """Compute :class:`HandFeatures` for ``hole`` on ``board``."""
    h, b = codes(hole), codes(board)
    if len(h) != 2 or len(set(h + b)) != len(h + b) or len(b) not in (0, 3, 4, 5):
        raise ValueError("need 2 hole cards and a 0/3/4/5-card board, all distinct")
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(b)]
    texture = board_texture(b)
    if b:
        cat = evaluate_best_codes(h + b)[0]
        strength, ahead, _ = _strength_vs_all(tuple(sorted(h)), tuple(sorted(b)))
        fb, nfb, sb = _blockers(h, b)
    else:
        cat, strength, ahead, fb, nfb, sb = -1, float("nan"), -1, False, False, 0.0
    spr = None
    if pot is not None and effective_stack is not None and pot > 0:
        spr = effective_stack / pot
    return HandFeatures(
        street=street,
        preflop_class=preflop_class(h),
        hand_category=cat,
        equity=hand_equity(h, b, equity_samples),
        current_strength=strength,
        combos_ahead=ahead,
        is_nuts=bool(b) and ahead == 0,
        draw=draw_type(h, b),
        flush_blocker=fb,
        nut_flush_blocker=nfb,
        straight_blocker_share=sb,
        texture=texture,
        spr=spr,
        position=position,
        effective_stack=effective_stack,
    )


def bucketize(value: float, edges: Sequence[float]) -> int:
    """Index of the half-open bucket ``[edges[i-1], edges[i])`` holding value."""
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), value,
                               side="right"))
