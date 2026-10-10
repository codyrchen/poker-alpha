"""Combo-level weighted hand ranges.

A :class:`WeightedRange` is a non-negative weight on each of the 1,326
two-card combinations, stored as one NumPy vector indexed by
:data:`COMBOS` (all ``(a, b)`` with ``a < b`` in ``itertools.combinations``
order). Removing blocked combos is a vector mask, so card removal is exact.

A range is a *belief* about an opponent's holding, never ground truth; the
:meth:`WeightedRange.entropy_bits` and :meth:`WeightedRange.effective_combos`
make its uncertainty visible.

Range notation accepted by :func:`parse_range`::

    AA, AKs, AKo, AK (= AKs + AKo), TT+ (TT..AA), A2s+ (A2s..AKs),
    KTo+ (KTo..KQo), 22-66 (pair span), A5s-A2s (kicker span),
    any item with :weight, e.g. "AA,KK:0.5,AQs+:0.25"
    random / any / 100% (all combos)
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cards import RANK_CHARS, codes
from .evaluator import evaluate_best_codes

COMBOS: np.ndarray = np.array(list(itertools.combinations(range(52), 2)),
                              dtype=np.int64)
NUM_COMBOS = len(COMBOS)  # 1326
COMBO_INDEX: Dict[Tuple[int, int], int] = {
    (int(a), int(b)): i for i, (a, b) in enumerate(COMBOS)}
# CARD_MASK[c] is True for every combo containing card c.
CARD_MASK = np.zeros((52, NUM_COMBOS), dtype=bool)
CARD_MASK[COMBOS[:, 0], np.arange(NUM_COMBOS)] = True
CARD_MASK[COMBOS[:, 1], np.arange(NUM_COMBOS)] = True


def combo_index(cards: Sequence) -> int:
    a, b = codes(cards)
    if a == b:
        raise ValueError("a combo needs two different cards")
    return COMBO_INDEX[(min(a, b), max(a, b))]


def _class_of(i: int) -> str:
    a, b = COMBOS[i]
    ra, rb = a % 13, b % 13
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANK_CHARS[hi] * 2
    return f"{RANK_CHARS[hi]}{RANK_CHARS[lo]}{'s' if a // 13 == b // 13 else 'o'}"


COMBO_CLASS: Tuple[str, ...] = tuple(_class_of(i) for i in range(NUM_COMBOS))
_CLASS_MEMBERS: Dict[str, np.ndarray] = {}
for _i, _c in enumerate(COMBO_CLASS):
    _CLASS_MEMBERS.setdefault(_c, []).append(_i)  # type: ignore[arg-type]
CLASS_MEMBERS: Dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.int64) for k, v in _CLASS_MEMBERS.items()}
del _CLASS_MEMBERS


def combo_cdf(probs: np.ndarray) -> np.ndarray:
    """Cumulative distribution for :func:`draw_combo` (computed once).

    Uses the same arithmetic as ``Generator.choice(n, p=probs)`` (cumsum,
    then divide by the last element) so draws are bit-identical to it.
    """
    cdf = np.asarray(probs, dtype=np.float64).cumsum()
    cdf /= cdf[-1]
    return cdf


def draw_combo(rng: np.random.Generator, cdf: np.ndarray) -> int:
    """One combo index; identical to ``rng.choice(len(cdf), p=probs)``
    (one ``rng.random()`` draw and a right-sided search) but avoids
    rebuilding the CDF on every call — ~15x faster for 1,326 combos."""
    return int(cdf.searchsorted(rng.random(), side="right"))


def blocked_mask(cards: Iterable) -> np.ndarray:
    """Boolean vector: combos that contain any of ``cards``."""
    cs = codes(cards)
    if not cs:
        return np.zeros(NUM_COMBOS, dtype=bool)
    return CARD_MASK[cs].any(axis=0)


class WeightedRange:
    """Non-negative weights over the 1,326 combos (immutable in use: every
    operation returns a new range)."""

    __slots__ = ("weights",)

    def __init__(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (NUM_COMBOS,):
            raise ValueError(f"weights must have shape ({NUM_COMBOS},)")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        self.weights = w

    # -- constructors ---------------------------------------------------------

    @classmethod
    def uniform(cls) -> "WeightedRange":
        return cls(np.ones(NUM_COMBOS))

    @classmethod
    def empty(cls) -> "WeightedRange":
        return cls(np.zeros(NUM_COMBOS))

    @classmethod
    def from_classes(cls, weights: Dict[str, float]) -> "WeightedRange":
        w = np.zeros(NUM_COMBOS)
        for name, weight in weights.items():
            if name not in CLASS_MEMBERS:
                raise ValueError(f"unknown hand class {name!r}")
            w[CLASS_MEMBERS[name]] = float(weight)
        return cls(w)

    @classmethod
    def from_string(cls, text: str) -> "WeightedRange":
        return cls.from_classes(parse_range(text))

    @classmethod
    def from_combos(cls, combos: Iterable[Tuple[Sequence, float]]) -> "WeightedRange":
        w = np.zeros(NUM_COMBOS)
        for cards, weight in combos:
            w[combo_index(cards)] += float(weight)
        return cls(w)

    # -- basic queries --------------------------------------------------------

    @property
    def total(self) -> float:
        return float(self.weights.sum())

    @property
    def num_live_combos(self) -> int:
        return int(np.count_nonzero(self.weights))

    def is_empty(self) -> bool:
        return self.total <= 0.0

    def probabilities(self) -> np.ndarray:
        t = self.total
        if t <= 0:
            raise ValueError("range is empty")
        return self.weights / t

    def normalize(self) -> "WeightedRange":
        return WeightedRange(self.probabilities())

    def weight_of(self, cards: Sequence) -> float:
        return float(self.weights[combo_index(cards)])

    def class_mass(self, cls: str) -> float:
        """Normalized probability mass of a hand class (e.g. ``"AKs"``)."""
        if cls not in CLASS_MEMBERS:
            raise ValueError(f"unknown hand class {cls!r}")
        return float(self.probabilities()[CLASS_MEMBERS[cls]].sum())

    def class_distribution(self) -> Dict[str, float]:
        p = self.probabilities()
        out = {c: float(p[idx].sum()) for c, idx in CLASS_MEMBERS.items()}
        return {c: v for c, v in out.items() if v > 0}

    def entropy_bits(self) -> float:
        p = self.probabilities()
        q = p[p > 0]
        return float(-(q * np.log2(q)).sum())

    def effective_combos(self) -> float:
        """``2 ** entropy``: how many equally likely combos this is worth."""
        return float(2.0 ** self.entropy_bits())

    # -- conditioning ---------------------------------------------------------

    def remove_cards(self, cards: Iterable) -> "WeightedRange":
        """Zero every combo using a known (dead/blocked) card."""
        w = self.weights.copy()
        w[blocked_mask(cards)] = 0.0
        return WeightedRange(w)

    def condition_on_board(self, board: Iterable) -> "WeightedRange":
        return self.remove_cards(board)

    def condition_on_known_cards(self, hero: Iterable = (), board: Iterable = (),
                                 dead: Iterable = ()) -> "WeightedRange":
        return self.remove_cards(list(codes(hero)) + list(codes(board))
                                 + list(codes(dead)))

    def update(self, likelihood: np.ndarray) -> "WeightedRange":
        """Bayes: ``posterior ∝ likelihood × prior`` (combo-wise).

        ``likelihood[i] = P(observation | combo i)``. Raises if the
        observation is impossible under every combo in the range (a model
        failure that callers must handle, not hide).
        """
        lik = np.asarray(likelihood, dtype=np.float64)
        if lik.shape != (NUM_COMBOS,) or np.any(lik < 0) or not np.all(np.isfinite(lik)):
            raise ValueError("likelihood must be a finite non-negative vector")
        post = self.weights * lik
        if post.sum() <= 0:
            raise ValueError("observation has zero likelihood under the range")
        return WeightedRange(post / post.sum())

    def scale(self, factor: float) -> "WeightedRange":
        return WeightedRange(self.weights * float(factor))

    def mix(self, other: "WeightedRange", alpha: float) -> "WeightedRange":
        """``(1 - alpha) * self + alpha * other`` on normalized ranges."""
        return WeightedRange((1 - alpha) * self.probabilities()
                             + alpha * other.probabilities())

    # -- sampling -------------------------------------------------------------

    def sample(self, rng: np.random.Generator, n: int = 1,
               exclude: Iterable = ()) -> List[Tuple[int, int]]:
        """Draw ``n`` combos (with replacement) avoiding ``exclude`` cards."""
        r = self.remove_cards(exclude) if exclude else self
        p = r.probabilities()
        idx = rng.choice(NUM_COMBOS, size=n, p=p)
        return [(int(COMBOS[i, 0]), int(COMBOS[i, 1])) for i in idx]

    def top_classes(self, k: int = 10) -> List[Tuple[str, float]]:
        dist = self.class_distribution()
        return sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def __repr__(self) -> str:
        if self.is_empty():
            return "WeightedRange(empty)"
        return (f"WeightedRange(live={self.num_live_combos}, "
                f"eff={self.effective_combos():.0f})")


# -- notation -------------------------------------------------------------------

def _rank(ch: str) -> int:
    try:
        return RANK_CHARS.index(ch.upper())
    except ValueError:
        raise ValueError(f"bad rank {ch!r}") from None


def _expand(token: str) -> List[str]:
    token = token.strip()
    if token.lower() in ("random", "any", "100%"):
        return list(CLASS_MEMBERS)
    if "-" in token:
        lo_t, hi_t = token.split("-")
        a, b = _expand_one(lo_t), _expand_one(hi_t)
        (ka, ra, sa), (kb, rb, sb) = a, b
        if ka == "pair" and kb == "pair":
            lo, hi = sorted((ra[0], rb[0]))
            return [RANK_CHARS[r] * 2 for r in range(lo, hi + 1)]
        if ka == kb == "nonpair" and ra[0] == rb[0] and sa == sb:
            lo, hi = sorted((ra[1], rb[1]))
            return _nonpair(ra[0], range(lo, hi + 1), sa)
        raise ValueError(f"bad range span {token!r}")
    plus = token.endswith("+")
    kind, ranks, suit = _expand_one(token.rstrip("+"))
    if kind == "pair":
        r = ranks[0]
        return [RANK_CHARS[x] * 2 for x in (range(r, 13) if plus else [r])]
    hi, lo = ranks
    kickers = range(lo, hi) if plus else [lo]
    return _nonpair(hi, kickers, suit)


def _nonpair(hi: int, kickers, suit: str) -> List[str]:
    out = []
    for k in kickers:
        base = f"{RANK_CHARS[hi]}{RANK_CHARS[k]}"
        if suit in ("s", ""):
            out.append(base + "s")
        if suit in ("o", ""):
            out.append(base + "o")
    return out


def _expand_one(tok: str):
    tok = tok.strip()
    if len(tok) == 2 and tok[0].upper() == tok[1].upper():
        return "pair", (_rank(tok[0]),), ""
    if len(tok) in (2, 3):
        a, b = _rank(tok[0]), _rank(tok[1])
        if a == b:
            raise ValueError(f"bad hand {tok!r}")
        suit = tok[2].lower() if len(tok) == 3 else ""
        if suit not in ("", "s", "o"):
            raise ValueError(f"bad suitedness in {tok!r}")
        return "nonpair", (max(a, b), min(a, b)), suit
    raise ValueError(f"cannot parse hand {tok!r}")


def parse_range(text: str) -> Dict[str, float]:
    """Parse range notation into ``{class: weight}`` (see module docstring)."""
    out: Dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        weight = 1.0
        if ":" in item:
            item, w = item.split(":")
            weight = float(w)
            if weight < 0:
                raise ValueError("weights must be non-negative")
        for cls in _expand(item):
            out[cls] = weight
    return out


# -- hand strength on a board ---------------------------------------------------

def strength_percentiles(board: Sequence, dead: Iterable = ()) -> np.ndarray:
    """For each combo, the share of *other live combos* its made hand beats
    on ``board`` (ties count half). Blocked combos get NaN.

    Needs a 3-5 card board. Each combo is evaluated once (1,326 evaluations),
    so this is cheap enough for per-decision range updates. It measures
    *current* strength only — draws are not credited.
    """
    b = codes(board)
    if not 3 <= len(b) <= 5:
        raise ValueError("strength percentiles need a 3-5 card board")
    blocked = blocked_mask(list(b) + list(codes(dead)))
    live = np.flatnonzero(~blocked)
    values = [evaluate_best_codes([int(COMBOS[i, 0]), int(COMBOS[i, 1])] + b)
              for i in live]
    # Rank values; ties share the average rank.
    order = sorted(range(len(live)), key=lambda j: values[j])
    ranks = np.empty(len(live))
    j = 0
    while j < len(order):
        k = j
        while k + 1 < len(order) and values[order[k + 1]] == values[order[j]]:
            k += 1
        avg = (j + k) / 2.0
        for m in range(j, k + 1):
            ranks[order[m]] = avg
        j = k + 1
    out = np.full(NUM_COMBOS, np.nan)
    denom = max(len(live) - 1, 1)
    out[live] = ranks / denom
    return out


_PREFLOP_STRENGTH: Optional[np.ndarray] = None


def preflop_strength() -> np.ndarray:
    """Per-combo preflop strength in [0, 1]: the 169-class equity vs a random
    hand (deterministic seeded estimate), rescaled to its min..max."""
    global _PREFLOP_STRENGTH
    if _PREFLOP_STRENGTH is None:
        from ..abstraction.cards import combos_for_class, hand_equity

        eq = np.zeros(NUM_COMBOS)
        for cls, idx in CLASS_MEMBERS.items():
            eq[idx] = hand_equity(combos_for_class(cls)[0], (), 600)
        lo, hi = eq.min(), eq.max()
        _PREFLOP_STRENGTH = (eq - lo) / (hi - lo)
    return _PREFLOP_STRENGTH
