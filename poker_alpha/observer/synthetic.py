"""Deterministic synthetic table renderer for tests and demos.

Renders a *PokerNow-style* table (generic felt, seat plates, light card
faces with a four-colour deck, chip text, dealer button, action highlight)
from a :class:`SyntheticTable` description, using the same calibration the
observer reads. It is a stand-in for real screenshots: recognition accuracy
measured on these images says nothing about accuracy on a real site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import TableCalibration
from .errors import require_pil
from .regions import Box


@dataclass
class SyntheticSeat:
    name: str = ""
    stack: Optional[float] = None       # None => empty seat
    bet: float = 0.0
    in_hand: bool = True
    all_in: bool = False


@dataclass
class SyntheticTable:
    seats: List[SyntheticSeat]
    dealer: int
    hero_cards: Tuple[str, ...] = ()
    board: Tuple[str, ...] = ()
    pot: float = 0.0
    actor: Optional[int] = None


def _font(px: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=max(8, int(px)))


def _fmt(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _text_in(draw, box: Box, text: str, color, height_frac: float = 0.7):
    l, t, r, b = box
    px = (b - t) * height_frac
    while True:  # shrink to fit the box width, like a real UI would
        font = _font(px)
        bb = draw.textbbox((0, 0), text, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= (r - l) * 0.95 or px <= 8:
            break
        px *= 0.9
    draw.text((l + (r - l - w) / 2 - bb[0], t + (b - t - h) / 2 - bb[1]),
              text, fill=tuple(color), font=font)


def draw_card(draw, box: Box, card: str, cal: TableCalibration) -> None:
    l, t, r, b = box
    draw.rounded_rectangle(box, radius=max(2, (r - l) // 8),
                           fill=(250, 250, 250), outline=(120, 120, 120))
    rank, suit = card[0], card[1]
    color = cal.suit_colors[suit]
    h = b - t
    _text_in(draw, (l, t + int(0.04 * h), r, t + int(0.48 * h)), rank, color, 0.8)
    # Suit mark: a simple filled shape per suit in the suit colour.
    cx, cy = (l + r) / 2, t + 0.72 * h
    s = min(r - l, h) * 0.18
    if suit == "h":
        draw.ellipse((cx - s, cy - s, cx, cy), fill=color)
        draw.ellipse((cx, cy - s, cx + s, cy), fill=color)
        draw.polygon([(cx - s, cy - s / 2), (cx + s, cy - s / 2), (cx, cy + s)], fill=color)
    elif suit == "d":
        draw.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=color)
    elif suit == "c":
        for dx, dy in ((0, -s / 2), (-s / 2, s / 6), (s / 2, s / 6)):
            draw.ellipse((cx + dx - s / 2, cy + dy - s / 2, cx + dx + s / 2,
                          cy + dy + s / 2), fill=color)
    else:
        draw.polygon([(cx, cy - s), (cx + s, cy + s / 3), (cx - s, cy + s / 3)], fill=color)
        draw.rectangle((cx - s / 6, cy, cx + s / 6, cy + s), fill=color)


def render_table(table: SyntheticTable, cal: TableCalibration,
                 size: Tuple[int, int] = (1280, 800),
                 margin: float = 0.06, noise: float = 0.0, seed: int = 0):
    """Render ``table``. ``noise`` adds Gaussian pixel noise (std, 0-255)."""
    require_pil()
    from PIL import Image, ImageDraw

    W, H = size
    img = Image.new("RGB", size, (24, 26, 30))
    d = ImageDraw.Draw(img)
    tb = (int(W * margin), int(H * margin), int(W * (1 - margin)), int(H * (1 - margin)))
    d.rectangle(tb, fill=tuple(cal.felt_color))

    def px(name: str) -> Box:
        return cal.regions[name].to_pixels(tb)

    if table.pot > 0:
        _text_in(d, px("pot"), _fmt(table.pot), cal.text_color)
    for i, card in enumerate(table.board):
        draw_card(d, px(f"board_{i}"), card, cal)
    for i, seat in enumerate(table.seats):
        if seat.stack is None:
            continue
        if f"seat{i}_name" in cal.regions and seat.name:
            _text_in(d, px(f"seat{i}_name"), seat.name, (200, 200, 200))
        _text_in(d, px(f"seat{i}_stack"),
                 "ALLIN" if seat.all_in else _fmt(seat.stack), cal.text_color)
        if seat.bet > 0:
            _text_in(d, px(f"seat{i}_bet"), _fmt(seat.bet), cal.text_color)
        if seat.in_hand and i != cal.hero_seat:
            l, t, r, b = px(f"seat{i}_cards")
            mid = (l + r) // 2
            d.rectangle((l, t, mid - 1, b), fill=tuple(cal.card_back_color))
            d.rectangle((mid + 1, t, r, b), fill=tuple(cal.card_back_color))
        if i == table.dealer:
            l, t, r, b = px(f"seat{i}_dealer")
            d.ellipse((l, t, r, b), fill=tuple(cal.button_color))
        if table.actor == i:
            d.rectangle(px(f"seat{i}_active"), fill=tuple(cal.highlight_color))
    for i, card in enumerate(table.hero_cards):
        draw_card(d, px(f"hero_card_{i}"), card, cal)
    if noise > 0:
        arr = np.asarray(img, dtype=np.float64)
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def random_table(rng: np.random.Generator, num_seats: int, hero_seat: int = 0,
                 street_cards: Optional[int] = None) -> SyntheticTable:
    """A random but internally consistent table for accuracy measurement."""
    deck = [r + s for r in "23456789TJQKA" for s in "shdc"]
    order = rng.permutation(len(deck))
    cards = [deck[i] for i in order]
    n_board = int(rng.choice([0, 3, 4, 5])) if street_cards is None else street_cards
    seats = []
    for i in range(num_seats):
        if i != hero_seat and rng.random() < 0.15:
            seats.append(SyntheticSeat(stack=None))
            continue
        stack = float(rng.choice([rng.integers(1, 300), rng.integers(1, 3000) / 2,
                                  rng.integers(100, 2_000_000) / 4]))
        bet = float(rng.choice([0, 0, rng.integers(1, 200) / 2]))
        seats.append(SyntheticSeat(name=f"p{i}", stack=stack, bet=bet,
                                   in_hand=bool(i == hero_seat or rng.random() < 0.6),
                                   all_in=bool(rng.random() < 0.05)))
    occupied = [i for i, s in enumerate(seats) if s.stack is not None]
    return SyntheticTable(
        seats=seats, dealer=int(rng.choice(occupied)),
        hero_cards=tuple(cards[:2]), board=tuple(cards[2:2 + n_board]),
        pot=float(rng.integers(3, 4000) / 2),
        actor=int(rng.choice(occupied)) if rng.random() < 0.7 else None)
