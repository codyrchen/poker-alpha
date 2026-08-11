"""Card, Deck, and Hand primitives for the 52-card engine.

A card is fundamentally an integer ``code`` in ``0..51``:

* ``rank = code % 13`` — 0 = deuce ... 8 = ten, 9 = jack, 10 = queen,
  11 = king, 12 = ace,
* ``suit = code // 13`` — 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades.

The hot paths (hand evaluation, Monte Carlo equity) work directly on these
integer codes; the :class:`Card` dataclass is a thin, hashable wrapper for
ergonomic construction and display, and converts to/from codes freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

NUM_CARDS = 52


def card_code(text: str) -> int:
    """Parse a card like ``"As"`` or ``"Td"`` into its integer code."""
    if len(text) != 2:
        raise ValueError(f"card string must be 2 chars, got {text!r}")
    try:
        rank = RANK_CHARS.index(text[0].upper())
        suit = SUIT_CHARS.index(text[1].lower())
    except ValueError:
        raise ValueError(f"bad card string {text!r}") from None
    return suit * 13 + rank


def card_str(code: int) -> str:
    """Format an integer card code as e.g. ``"As"``."""
    return RANK_CHARS[code % 13] + SUIT_CHARS[code // 13]


def codes(cards: Iterable["Card | int | str"]) -> List[int]:
    """Normalize a mixed iterable of cards to integer codes."""
    out: List[int] = []
    for c in cards:
        if isinstance(c, Card):
            out.append(c.code)
        elif isinstance(c, int):
            if not 0 <= c < NUM_CARDS:
                raise ValueError(f"card code out of range: {c}")
            out.append(c)
        elif isinstance(c, str):
            out.append(card_code(c))
        else:
            raise TypeError(f"not a card: {c!r}")
    return out


@dataclass(frozen=True, order=True)
class Card:
    """A single playing card, wrapping an integer ``code`` in ``0..51``."""

    code: int

    def __post_init__(self) -> None:
        if not 0 <= self.code < NUM_CARDS:
            raise ValueError(f"card code out of range: {self.code}")

    @classmethod
    def from_str(cls, text: str) -> "Card":
        return cls(card_code(text))

    @property
    def rank(self) -> int:
        return self.code % 13

    @property
    def suit(self) -> int:
        return self.code // 13

    def __str__(self) -> str:
        return card_str(self.code)

    def __repr__(self) -> str:
        return f"Card({card_str(self.code)!r})"


class Deck:
    """A 52-card deck with seeded shuffling and dead-card removal."""

    def __init__(self, rng: Optional[np.random.Generator] = None,
                 exclude: Iterable["Card | int | str"] = ()) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        dead = set(codes(exclude))
        self.cards: List[int] = [c for c in range(NUM_CARDS) if c not in dead]

    def __len__(self) -> int:
        return len(self.cards)

    def shuffle(self) -> None:
        self.rng.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[int]:
        """Remove and return ``n`` card codes from the top of the deck."""
        if n > len(self.cards):
            raise ValueError(f"cannot draw {n} from deck of {len(self.cards)}")
        drawn, self.cards = self.cards[:n], self.cards[n:]
        return drawn

    def sample(self, n: int) -> List[int]:
        """Return ``n`` distinct card codes without mutating the deck."""
        idx = self.rng.choice(len(self.cards), size=n, replace=False)
        return [self.cards[i] for i in idx]


@dataclass(frozen=True)
class Hand:
    """An ordered collection of cards (hole cards, or hole + board)."""

    cards: tuple

    @classmethod
    def from_strs(cls, *texts: str) -> "Hand":
        return cls(tuple(card_code(t) for t in texts))

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return " ".join(card_str(c) for c in self.cards)
