"""Ground-truth annotations for real screenshots and field-level scoring.

Annotation format (``pokeralpha.screenshot_annotation/v1``), one JSON file per
screenshot in ``<fixture-dir>/annotations/<image stem>.json``::

    {
      "format": "pokeralpha.screenshot_annotation/v1",   (optional)
      "image": "raw/hand_0001.png",                       (optional; default raw/<stem>.png)
      "num_seats": 6,
      "hero_seat": 3,
      "dealer_seat": 3,
      "hero_cards": ["As", "Kd"],
      "board": ["Qs", "Jh", "4c"],
      "pot": 13.5,
      "street": "flop",                                   (optional; checked vs board length)
      "seats": [
        {"seat": 0, "stack": 97.5, "bet": 2.5, "active": true,
         "all_in": false, "name": "alice"},               (all_in, name optional)
        ...
      ]
    }

:func:`validate_annotation` checks an annotation dict for internal
consistency (card syntax, duplicates, board length vs street, seat ranges,
hero occupied, non-negative amounts) before it is used as ground truth.

``seats`` lists occupied seats only; seats absent from the list are empty.
``active`` means still in the hand. Amounts are in the units the client
displays. Nothing here is real data until a human annotates real images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ANNOTATION_FORMAT = "pokeralpha.screenshot_annotation/v1"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class SeatTruth:
    seat: int
    stack: Optional[float]
    bet: float
    active: bool
    all_in: bool = False
    name: str = ""


STREETS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def validate_annotation(d: dict) -> List[str]:
    """Problems that make an annotation unusable as ground truth ([] = ok)."""
    from ..poker.cards import card_code

    errs: List[str] = []
    if d.get("format", ANNOTATION_FORMAT) != ANNOTATION_FORMAT:
        errs.append(f"unsupported format {d.get('format')!r}")
    n = d.get("num_seats")
    if not isinstance(n, int) or not 2 <= n <= 10:
        errs.append("num_seats must be an integer in 2..10")
        n = 10
    for k in ("hero_seat", "dealer_seat"):
        v = d.get(k)
        if v is not None and (not isinstance(v, int) or not 0 <= v < n):
            errs.append(f"{k} out of range")
    cards = list(d.get("hero_cards") or []) + list(d.get("board") or [])
    for c in cards:
        try:
            card_code(c)
        except (ValueError, KeyError, IndexError, TypeError):
            errs.append(f"bad card {c!r}")
    if len(set(cards)) != len(cards):
        errs.append("duplicate cards")
    if d.get("hero_cards") and len(d["hero_cards"]) != 2:
        errs.append("hero_cards must have 2 cards")
    blen = len(d.get("board") or [])
    if blen not in (0, 3, 4, 5):
        errs.append("board must have 0, 3, 4 or 5 cards")
    if "street" in d and STREETS.get(d["street"]) != blen:
        errs.append(f"street {d['street']!r} inconsistent with {blen} board cards")
    seats = d.get("seats")
    if not isinstance(seats, list) or not seats:
        errs.append("seats must be a non-empty list")
        seats = []
    seen = set()
    for s in seats:
        i = s.get("seat")
        if not isinstance(i, int) or not 0 <= i < n:
            errs.append(f"seat index {i!r} out of range")
        if i in seen:
            errs.append(f"seat {i} listed twice")
        seen.add(i)
        for k in ("stack", "bet"):
            v = s.get(k)
            if v is not None and (not isinstance(v, (int, float)) or v < 0):
                errs.append(f"seat {i} {k} must be a non-negative number")
        if s.get("all_in") and s.get("stack") not in (None, 0, 0.0):
            errs.append(f"seat {i} all_in with non-zero stack")
    if d.get("hero_seat") is not None and seats and d["hero_seat"] not in seen:
        errs.append("hero seat not listed as occupied")
    pot = d.get("pot")
    if pot is not None and (not isinstance(pot, (int, float)) or pot < 0):
        errs.append("pot must be a non-negative number")
    return errs


@dataclass(frozen=True)
class Annotation:
    name: str
    image: Path
    num_seats: int
    hero_seat: int
    dealer_seat: Optional[int]
    hero_cards: tuple
    board: tuple
    pot: Optional[float]
    seats: Dict[int, SeatTruth]


def load_annotation(path: Path, fixture_dir: Path) -> Annotation:
    d = json.loads(Path(path).read_text())
    problems = validate_annotation(d)
    if problems:
        raise ValueError(f"{path}: invalid annotation: {'; '.join(problems)}")
    stem = Path(path).stem
    if "image" in d:
        image = fixture_dir / d["image"]
    else:
        cands = [fixture_dir / "raw" / (stem + s) for s in IMAGE_SUFFIXES]
        image = next((c for c in cands if c.exists()), cands[0])
    seats = {}
    for s in d["seats"]:
        seats[int(s["seat"])] = SeatTruth(int(s["seat"]),
                                          None if s.get("stack") is None else float(s["stack"]),
                                          float(s.get("bet", 0.0)),
                                          bool(s.get("active", True)),
                                          bool(s.get("all_in", False)), str(s.get("name", "")))
    return Annotation(stem, image, int(d["num_seats"]), int(d["hero_seat"]),
                      None if d.get("dealer_seat") is None else int(d["dealer_seat"]),
                      tuple(d.get("hero_cards") or ()), tuple(d.get("board") or ()),
                      None if d.get("pot") is None else float(d["pot"]), seats)


def load_fixture_dir(fixture_dir: Path) -> List[Annotation]:
    ann_dir = Path(fixture_dir) / "annotations"
    if not ann_dir.is_dir():
        return []
    return [load_annotation(p, Path(fixture_dir)) for p in sorted(ann_dir.glob("*.json"))]


@dataclass
class FieldTally:
    correct: int = 0
    total: int = 0
    abs_errors: List[float] = field(default_factory=list)
    unreadable: int = 0

    def add(self, ok: bool, err: Optional[float] = None, unreadable: bool = False):
        self.total += 1
        self.correct += bool(ok)
        if err is not None:
            self.abs_errors.append(err)
        self.unreadable += bool(unreadable)

    def summary(self) -> dict:
        out = {"correct": self.correct, "total": self.total,
               "accuracy": self.correct / self.total if self.total else None}
        if self.abs_errors or self.unreadable:
            out["mae"] = (sum(self.abs_errors) / len(self.abs_errors)
                          if self.abs_errors else None)
            out["unreadable"] = self.unreadable
        return out


def _amount(tally: FieldTally, read, truth: Optional[float]) -> bool:
    if truth is None:
        return True
    if read is None or isinstance(read, str):
        tally.add(False, unreadable=True)
        return False
    ok = abs(read - truth) < 1e-6
    tally.add(ok, abs(read - truth))
    return ok


def score(frames) -> dict:
    """Field-level accuracy over ``(FrameObservation, Annotation)`` pairs."""
    keys = ("hero_cards", "board_cards", "stack", "bet", "pot", "dealer",
            "seat_occupancy", "street")
    t = {k: FieldTally() for k in keys}
    hero_pairs = FieldTally()
    board_sets = FieldTally()
    calib: List[tuple] = []          # (confidence, correct) per card/amount reading
    full_ok = 0
    n = 0
    for obs, ann in frames:
        n += 1
        ok = True
        pair_ok = True
        for i in range(2):
            want = ann.hero_cards[i] if i < len(ann.hero_cards) else None
            good = obs.value(f"hero_card_{i}") == want
            t["hero_cards"].add(good)
            calib.append((obs.confidence(f"hero_card_{i}"), good))
            pair_ok &= good
        hero_pairs.add(pair_ok)
        ok &= pair_ok
        board_ok = True
        for i in range(5):
            want = ann.board[i] if i < len(ann.board) else None
            good = obs.value(f"board_{i}") == want
            t["board_cards"].add(good)
            calib.append((obs.confidence(f"board_{i}"), good))
            board_ok &= good
        board_sets.add(board_ok)
        ok &= board_ok
        read_len = sum(1 for i in range(5) if obs.value(f"board_{i}") is not None)
        good = read_len == len(ann.board)
        t["street"].add(good)
        ok &= good
        ok &= _amount(t["pot"], obs.value("pot"), ann.pot)
        if ann.dealer_seat is not None:
            good = obs.value("dealer") == ann.dealer_seat
            t["dealer"].add(good)
            ok &= good
        for s in range(ann.num_seats):
            truth = ann.seats.get(s)
            good = obs.value(f"seat{s}.occupied") == (truth is not None)
            t["seat_occupancy"].add(good)
            ok &= good
            if truth is None:
                continue
            ok &= _amount(t["stack"], obs.value(f"seat{s}.stack"), truth.stack)
            ok &= _amount(t["bet"], obs.value(f"seat{s}.bet"), truth.bet)
        full_ok += ok
    out = {k: v.summary() for k, v in t.items()}
    out["hero_cards_exact_pair"] = hero_pairs.summary()
    out["board_exact"] = board_sets.summary()
    out["confidence_calibration"] = _calibration(calib)
    out["full_state"] = {"correct": full_ok, "total": n,
                         "accuracy": full_ok / n if n else None}
    out["screenshots"] = n
    return out


def _calibration(pairs, edges=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01)):
    """Accuracy of card readings by reported confidence bin."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [ok for c, ok in pairs if lo <= c < hi]
        if sel:
            out.append({"confidence": [lo, min(hi, 1.0)], "n": len(sel),
                        "accuracy": sum(sel) / len(sel)})
    return out
