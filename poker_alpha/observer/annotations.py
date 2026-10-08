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
      "seats": [
        {"seat": 0, "stack": 97.5, "bet": 2.5, "active": true},
        ...
      ]
    }

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
    fmt = d.get("format", ANNOTATION_FORMAT)
    if fmt != ANNOTATION_FORMAT:
        raise ValueError(f"{path}: unsupported annotation format {fmt!r}")
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
                                          bool(s.get("active", True)))
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
            "seat_occupancy")
    t = {k: FieldTally() for k in keys}
    full_ok = 0
    n = 0
    for obs, ann in frames:
        n += 1
        ok = True
        for i in range(2):
            want = ann.hero_cards[i] if i < len(ann.hero_cards) else None
            good = obs.value(f"hero_card_{i}") == want
            t["hero_cards"].add(good)
            ok &= good
        for i in range(5):
            want = ann.board[i] if i < len(ann.board) else None
            good = obs.value(f"board_{i}") == want
            t["board_cards"].add(good)
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
    out["full_state"] = {"correct": full_ok, "total": n,
                         "accuracy": full_ok / n if n else None}
    out["screenshots"] = n
    return out
