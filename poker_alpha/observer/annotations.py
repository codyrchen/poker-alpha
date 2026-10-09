"""Ground-truth annotations for real screenshots and field-level scoring.

Annotation format (``pokeralpha.screenshot_annotation/v1``), one JSON file per
screenshot in ``<fixture-dir>/annotations/<image stem>.json``::

    {
      "format": "pokeralpha.screenshot_annotation/v1",   (optional)
      "image": "raw/hand_0001.png",                       (optional; default raw/<stem>.png)
      "role": "validation",           (tuning | validation | held_out; default unassigned)
      "status": "complete",           (unreviewed | partial | complete | skip; default complete)
      "annotated": ["hero_cards", "board", "pot", "seats"],   (optional, see below)
      "num_seats": 6,
      "hero_seat": 3,
      "dealer_seat": 3,
      "actor": 4,                     (seat to act; null = nobody)
      "hero_cards": ["As", "Kd"],
      "board": ["Qs", "Jh", "4c"],
      "pot": 13.5,                    (the pot number the client displays)
      "pot_total": 18.5,              (all chips in the middle incl. bets, if known)
      "street": "flop",               (optional; checked vs board length)
      "seats": [
        {"seat": 0, "stack": 97.5, "bet": 2.5, "active": true,
         "all_in": false, "name": "alice"},               (all_in, name optional)
        ...
      ],
      "notes": "free text"
    }

Two flavours, distinguished by the ``annotated`` key:

* **legacy / complete** (no ``annotated``): every field is ground truth;
  missing ``hero_cards`` / ``board`` mean "none", ``seats`` lists occupied
  seats only (absent = empty), missing ``bet`` = 0, ``active`` = true.
* **partial** (``annotated`` lists the groups that carry ground truth, from
  ``hero_cards board street dealer_seat actor pot pot_total seats``): any
  field outside the list, any seat not listed, and any seat key not given
  (``occupied``, ``stack``, ``bet``, ``active``, ``all_in``) is *unknown*
  and is not scored. Seat entries may say ``"occupied": false``.

``active`` means still in the hand (so folded = occupied and not active).
``status`` "unreviewed" and "skip" frames are never scored. ``role``
separates tuning data from validation / held-out data (Phase 45): real
accuracy is only ever claimed from validation / held-out frames.

:func:`validate_annotation` checks an annotation dict for internal
consistency before it is used as ground truth. Amounts are in the units the
client displays. Nothing here is real data until a human annotates real images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

ANNOTATION_FORMAT = "pokeralpha.screenshot_annotation/v1"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
ROLES = ("tuning", "validation", "held_out", "unassigned")
STATUSES = ("unreviewed", "partial", "complete", "skip")
GROUPS = ("hero_cards", "board", "street", "dealer_seat", "actor", "pot", "pot_total",
          "seats")
SEAT_KEYS = ("occupied", "stack", "bet", "active", "all_in")


@dataclass(frozen=True)
class SeatTruth:
    """Ground truth for one seat; ``None`` = unknown (not scored)."""

    seat: int
    stack: Optional[float]
    bet: Optional[float] = 0.0
    active: Optional[bool] = True
    all_in: Optional[bool] = False
    name: str = ""
    occupied: Optional[bool] = True

    @property
    def folded(self) -> Optional[bool]:
        if self.occupied is None or self.active is None:
            return None
        return bool(self.occupied and not self.active)


STREETS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def validate_annotation(d: dict) -> List[str]:
    """Problems that make an annotation unusable as ground truth ([] = ok)."""
    from ..poker.cards import card_code

    errs: List[str] = []
    if d.get("format", ANNOTATION_FORMAT) != ANNOTATION_FORMAT:
        errs.append(f"unsupported format {d.get('format')!r}")
    if d.get("role", "unassigned") not in ROLES:
        errs.append(f"role must be one of {ROLES}")
    if d.get("status", "complete") not in STATUSES:
        errs.append(f"status must be one of {STATUSES}")
    partial = "annotated" in d
    if partial:
        bad = set(d["annotated"] or ()) - set(GROUPS)
        if bad:
            errs.append(f"unknown annotated groups {sorted(bad)}")
    n = d.get("num_seats")
    if not isinstance(n, int) or not 2 <= n <= 10:
        errs.append("num_seats must be an integer in 2..10")
        n = 10
    for k in ("hero_seat", "dealer_seat", "actor"):
        v = d.get(k)
        if v is not None and (not isinstance(v, int) or not 0 <= v < n):
            errs.append(f"{k} out of range")
    if not isinstance(d.get("hero_seat"), int):
        errs.append("hero_seat is required")
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
    if d.get("street") is not None and (d["street"] not in STREETS or (
            ("board" in d or not partial) and STREETS.get(d["street"]) != blen)):
        errs.append(f"street {d['street']!r} inconsistent with {blen} board cards")
    seats = d.get("seats")
    if seats is None and partial:
        seats = []
    if not isinstance(seats, list) or (not seats and not partial):
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
        if s.get("occupied") is False and any(s.get(k) for k in ("stack", "bet", "active",
                                                                  "all_in")):
            errs.append(f"seat {i} empty but has chips / is in the hand")
    hero = d.get("hero_seat")
    if hero is not None and seats and not partial and hero not in seen:
        errs.append("hero seat not listed as occupied")
    for k in ("pot", "pot_total"):
        v = d.get(k)
        if v is not None and (not isinstance(v, (int, float)) or v < 0):
            errs.append(f"{k} must be a non-negative number")
    return errs


@dataclass(frozen=True)
class Annotation:
    name: str
    image: Path
    num_seats: int
    hero_seat: int
    dealer_seat: Optional[int]
    hero_cards: Optional[tuple]          # None = unknown
    board: Optional[tuple]               # None = unknown
    pot: Optional[float]
    seats: Dict[int, SeatTruth]          # every seat; unknown seats have occupied=None
    known: FrozenSet[str] = frozenset(GROUPS)
    actor: Optional[int] = None
    pot_total: Optional[float] = None
    street: Optional[str] = None
    role: str = "unassigned"
    status: str = "complete"
    notes: str = ""

    def knows(self, group: str) -> bool:
        return group in self.known

    @property
    def scored(self) -> bool:
        return self.status in ("partial", "complete")


def _opt_float(v):
    return None if v is None else float(v)


def _opt_bool(v):
    return None if v is None else bool(v)


def annotation_from_dict(d: dict, name: str, image: Path,
                         role: Optional[str] = None) -> Annotation:
    problems = validate_annotation(d)
    if problems:
        raise ValueError(f"{name}: invalid annotation: {'; '.join(problems)}")
    n = int(d["num_seats"])
    partial = "annotated" in d
    seats: Dict[int, SeatTruth] = {}
    if partial:
        known = frozenset(d["annotated"])
        for s in d.get("seats") or []:
            occ = s.get("occupied", True if any(k in s for k in ("stack", "bet")) else None)
            seats[int(s["seat"])] = SeatTruth(
                int(s["seat"]), _opt_float(s.get("stack")), _opt_float(s.get("bet")),
                _opt_bool(s.get("active")), _opt_bool(s.get("all_in")),
                str(s.get("name", "")), _opt_bool(occ))
        for i in range(n):
            seats.setdefault(i, SeatTruth(i, None, None, None, None, "", None))
    else:
        known = {"hero_cards", "board", "street", "pot", "seats"}
        if d.get("dealer_seat") is not None:
            known.add("dealer_seat")
        if d.get("pot") is None:
            known.discard("pot")
        for k in ("actor", "pot_total"):
            if k in d:
                known.add(k)
        known = frozenset(known)
        for s in d["seats"]:
            seats[int(s["seat"])] = SeatTruth(
                int(s["seat"]), _opt_float(s.get("stack")), float(s.get("bet", 0.0)),
                bool(s.get("active", True)), bool(s.get("all_in", False)),
                str(s.get("name", "")), True)
        for i in range(n):
            seats.setdefault(i, SeatTruth(i, None, 0.0, False, False, "", False))
    hero = d.get("hero_cards")
    board = d.get("board")
    if not partial:
        hero, board = tuple(hero or ()), tuple(board or ())
    else:
        hero = tuple(hero) if "hero_cards" in known and hero is not None else (
            () if "hero_cards" in known else None)
        board = tuple(board) if "board" in known and board is not None else (
            () if "board" in known else None)
    street = d.get("street")
    if street is None and board is not None:
        street = {v: k for k, v in STREETS.items()}.get(len(board))
    return Annotation(name, image, n, int(d["hero_seat"]),
                      None if d.get("dealer_seat") is None else int(d["dealer_seat"]),
                      hero, board, _opt_float(d.get("pot")), seats, known,
                      None if d.get("actor") is None else int(d["actor"]),
                      _opt_float(d.get("pot_total")), street,
                      role or d.get("role", "unassigned"), d.get("status", "complete"),
                      str(d.get("notes", "")))


def load_annotation(path: Path, fixture_dir: Path, role: Optional[str] = None) -> Annotation:
    d = json.loads(Path(path).read_text())
    stem = Path(path).stem
    if "image" in d:
        image = Path(fixture_dir) / d["image"]
    else:
        cands = [Path(fixture_dir) / "raw" / (stem + s) for s in IMAGE_SUFFIXES]
        image = next((c for c in cands if c.exists()), cands[0])
    return annotation_from_dict(d, stem, image, role)


def load_splits(fixture_dir: Path) -> Dict[str, str]:
    """Optional ``splits.json``: {"<annotation stem>": role}; overrides the
    per-annotation ``role``."""
    p = Path(fixture_dir) / "splits.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    bad = {k: v for k, v in d.items() if v not in ROLES}
    if bad:
        raise ValueError(f"{p}: unknown roles {bad}")
    return d


def load_fixture_dir(fixture_dir: Path, include_unscored: bool = False) -> List[Annotation]:
    """Annotations of a fixture (or session) directory. Frames marked
    "unreviewed" or "skip" are dropped unless ``include_unscored``."""
    ann_dir = Path(fixture_dir) / "annotations"
    if not ann_dir.is_dir():
        return []
    splits = load_splits(fixture_dir)
    out = [load_annotation(p, Path(fixture_dir), splits.get(p.stem))
           for p in sorted(ann_dir.glob("*.json"))]
    return out if include_unscored else [a for a in out if a.scored]


@dataclass
class FieldTally:
    correct: int = 0
    total: int = 0
    abs_errors: List[float] = field(default_factory=list)
    rel_errors: List[float] = field(default_factory=list)
    unreadable: int = 0

    def add(self, ok: bool, err: Optional[float] = None, unreadable: bool = False,
            rel: Optional[float] = None):
        self.total += 1
        self.correct += bool(ok)
        if err is not None:
            self.abs_errors.append(err)
        if rel is not None:
            self.rel_errors.append(rel)
        self.unreadable += bool(unreadable)

    def summary(self) -> dict:
        out = {"correct": self.correct, "total": self.total,
               "accuracy": self.correct / self.total if self.total else None}
        if self.abs_errors or self.unreadable:
            out["mae"] = (sum(self.abs_errors) / len(self.abs_errors)
                          if self.abs_errors else None)
            out["unreadable"] = self.unreadable
        if self.rel_errors:
            out["mean_relative_error"] = sum(self.rel_errors) / len(self.rel_errors)
        return out


def _amount(tally: FieldTally, read, truth: Optional[float], calib=None, conf=None) -> bool:
    if truth is None:
        return True
    if read is None or isinstance(read, str):
        tally.add(False, unreadable=True)
        ok = False
    else:
        ok = abs(read - truth) < 1e-6
        tally.add(ok, abs(read - truth),
                  rel=abs(read - truth) / truth if truth > 0 else None)
    if calib is not None:
        calib.append((conf, ok))
    return ok


def _check(tally: FieldTally, good: bool) -> bool:
    tally.add(good)
    return good


FRAME_METRICS = ("hero_card", "hero_exact_pair", "board_card", "board_exact", "card_rank",
                 "card_suit", "stack", "bet", "pot", "occupied", "in_hand", "folded",
                 "all_in", "dealer", "actor", "street", "full_state")


def score(frames) -> dict:
    """Frame-level accuracy over ``(FrameObservation, Annotation)`` pairs.

    Only fields the annotation knows are scored. Confidence values are the
    recognizers' match scores, not probabilities; the confidence table only
    shows how often readings in each score bucket were right.
    """
    t = {k: FieldTally() for k in FRAME_METRICS}
    calib_cards: List[tuple] = []
    calib_amounts: List[tuple] = []
    duplicate_frames = 0
    n = 0
    for obs, ann in frames:
        n += 1
        ok = True
        read_cards = [obs.value(f"hero_card_{i}") for i in range(2)] + \
            [obs.value(f"board_{i}") for i in range(5)]
        read_cards = [c for c in read_cards if c is not None]
        duplicate_frames += len(set(read_cards)) != len(read_cards)
        truth_cards = []
        if ann.hero_cards is not None:
            pair = True
            for i in range(2):
                want = ann.hero_cards[i] if i < len(ann.hero_cards) else None
                got = obs.value(f"hero_card_{i}")
                good = _check(t["hero_card"], got == want)
                calib_cards.append((obs.confidence(f"hero_card_{i}"), good))
                pair &= good
                if want is not None:
                    truth_cards.append((want, got))
            ok &= _check(t["hero_exact_pair"], pair)
        if ann.board is not None:
            exact = True
            for i in range(5):
                want = ann.board[i] if i < len(ann.board) else None
                got = obs.value(f"board_{i}")
                good = _check(t["board_card"], got == want)
                calib_cards.append((obs.confidence(f"board_{i}"), good))
                exact &= good
                if want is not None:
                    truth_cards.append((want, got))
            ok &= _check(t["board_exact"], exact)
        for want, got in truth_cards:
            t["card_rank"].add(got is not None and got[0] == want[0])
            t["card_suit"].add(got is not None and got[1] == want[1])
        if ann.street is not None:
            read_len = sum(1 for i in range(5) if obs.value(f"board_{i}") is not None)
            ok &= _check(t["street"], read_len == STREETS[ann.street])
        if ann.knows("pot"):
            ok &= _amount(t["pot"], obs.value("pot"), ann.pot, calib_amounts,
                          obs.confidence("pot"))
        if ann.knows("dealer_seat") and ann.dealer_seat is not None:
            ok &= _check(t["dealer"], obs.value("dealer") == ann.dealer_seat)
        if ann.knows("actor"):
            ok &= _check(t["actor"], obs.value("actor") == ann.actor)
        if ann.knows("seats"):
            for s in range(ann.num_seats):
                truth = ann.seats.get(s)
                if truth is None or truth.occupied is None:
                    continue
                read_occ = bool(obs.value(f"seat{s}.occupied"))
                ok &= _check(t["occupied"], read_occ == truth.occupied)
                if not truth.occupied:
                    continue
                read_in = bool(obs.value(f"seat{s}.in_hand"))
                if truth.active is not None:
                    ok &= _check(t["in_hand"], read_in == truth.active)
                    ok &= _check(t["folded"], (read_occ and not read_in) == truth.folded)
                if truth.all_in is not None:
                    ok &= _check(t["all_in"], bool(obs.value(f"seat{s}.all_in")) == truth.all_in)
                ok &= _amount(t["stack"], obs.value(f"seat{s}.stack"), truth.stack,
                              calib_amounts, obs.confidence(f"seat{s}.stack"))
                ok &= _amount(t["bet"], obs.value(f"seat{s}.bet"), truth.bet,
                              calib_amounts, obs.confidence(f"seat{s}.bet"))
        t["full_state"].add(ok)
    out = {k: v.summary() for k, v in t.items()}
    out["duplicate_card_frames"] = {"frames": duplicate_frames, "total": n}
    out["confidence_buckets"] = {"cards": confidence_buckets(calib_cards),
                                 "amounts": confidence_buckets(calib_amounts),
                                 "all": confidence_buckets(calib_cards + calib_amounts),
                                 "meaning": "recognizer match scores, NOT probabilities"}
    out["screenshots"] = n
    # backwards-compatible names used by earlier reports
    out["hero_cards"] = out["hero_card"]
    out["board_cards"] = out["board_card"]
    out["seat_occupancy"] = out["occupied"]
    out["hero_cards_exact_pair"] = out["hero_exact_pair"]
    out["confidence_calibration"] = out["confidence_buckets"]["all"]
    return out


def confidence_buckets(pairs, width: float = 0.1) -> List[dict]:
    """count / accuracy / error rate of readings per match-score bucket
    [0, 0.1), [0.1, 0.2), ... [0.9, 1.0]."""
    out = []
    k = int(round(1 / width))
    for i in range(k):
        lo, hi = i * width, (i + 1) * width
        sel = [ok for c, ok in pairs
               if c is not None and lo <= c < hi or (i == k - 1 and c is not None and c >= hi)]
        out.append({"bucket": [round(lo, 2), round(hi, 2)], "count": len(sel),
                    "accuracy": (sum(sel) / len(sel)) if sel else None,
                    "error_rate": (1 - sum(sel) / len(sel)) if sel else None})
    return out


REAL_VALIDATION_ROLES = ("validation", "held_out")


def score_by_role(frames) -> dict:
    """Metrics per dataset role, plus the only number that may be called
    real accuracy: validation + held-out frames combined.

    With no validation / held-out frames the real-validation status is
    ``"REAL VALIDATION: BLOCKED"`` (never 0% or 100%).
    """
    roles: Dict[str, list] = {r: [] for r in ROLES}
    for obs, ann in frames:
        roles.setdefault(ann.role, []).append((obs, ann))
    by_role = {r: {"frames": len(f), "metrics": score(f) if f else None}
               for r, f in roles.items()}
    real = [x for r in REAL_VALIDATION_ROLES for x in roles[r]]
    out = {"by_role": by_role}
    if real:
        out["real_validation"] = {
            "status": "MEASURED", "frames": len(real), "roles": list(REAL_VALIDATION_ROLES),
            "metrics": score(real)}
    else:
        out["real_validation"] = {
            "status": "REAL VALIDATION: BLOCKED", "frames": 0,
            "reason": "no annotated validation or held-out frames"}
    if roles["tuning"]:
        out["tuning_fit"] = {
            "frames": len(roles["tuning"]),
            "label": "TUNING-FIT ONLY: these frames were used to build / tune the "
                     "recognizers; not evidence of real accuracy"}
    if roles["unassigned"]:
        out["unassigned"] = {
            "frames": len(roles["unassigned"]),
            "label": "no dataset role: excluded from real-accuracy claims"}
    return out
