"""Annotating recorded observer sessions (Phase 43); Streamlit-free logic.

Annotations are written to ``<session>/annotations/<sample id>.json`` in the
``pokeralpha.screenshot_annotation/v1`` schema (partial flavour: only the
groups listed in ``annotated`` are ground truth), with ``"image":
"frames/<id>.png"``, so a session directory can be scored directly:

    python experiments/observer_validation.py --fixture-dir ~/pokeralpha_sessions/<id>

Statuses: ``unreviewed`` (default; never scored), ``partial``, ``complete``,
``skip`` (never scored). Prefilled values come from the observer's own
readings; they are a typing aid, must be checked by a human, and a
prefilled-but-unedited annotation stays ``unreviewed`` until saved with
another status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..poker.cards import card_code
from .annotations import (ANNOTATION_FORMAT, GROUPS, ROLES, STATUSES, STREETS,
                          validate_annotation)
from .session_replay import load_session

SEAT_STATUSES = ("unknown", "empty", "in hand", "folded", "all-in", "sitting out")


def parse_card(text: str) -> Optional[str]:
    """'Js' / 'js' / 'JS' / '10s' / 'T♠' -> 'Js' / 'Ts'; '' -> None."""
    t = (text or "").strip()
    if not t:
        return None
    for sym, s in (("♠", "s"), ("♥", "h"), ("♦", "d"), ("♣", "c")):
        t = t.replace(sym, s)
    if t[:2] == "10":
        t = "T" + t[2:]
    if len(t) != 2:
        raise ValueError(f"bad card {text!r}")
    card = t[0].upper() + t[1].lower()
    card_code(card)
    return card


def parse_cards(text: str) -> Optional[List[str]]:
    """'' -> None (unknown); '-' -> [] (none); 'Qs Jh 4c' -> cards."""
    t = (text or "").strip()
    if not t:
        return None
    if t in ("-", "none"):
        return []
    return [parse_card(x) for x in t.replace(",", " ").split()]


def parse_amount(text: str) -> Optional[float]:
    t = (text or "").strip().replace(",", "")
    if not t:
        return None
    v = float(t)
    if v < 0:
        raise ValueError("amount must be non-negative")
    return v


def seat_status_to_fields(status: str) -> Dict[str, Optional[bool]]:
    return {
        "unknown": {},
        "empty": {"occupied": False},
        "in hand": {"occupied": True, "active": True, "all_in": False},
        "folded": {"occupied": True, "active": False, "all_in": False},
        "all-in": {"occupied": True, "active": True, "all_in": True},
        "sitting out": {"occupied": True, "active": False},
    }[status]


def seat_fields_to_status(seat: dict) -> str:
    if not seat or seat.get("occupied") is None:
        return "unknown"
    if seat.get("occupied") is False:
        return "empty"
    if seat.get("all_in"):
        return "all-in"
    if seat.get("active") is True:
        return "in hand"
    if seat.get("active") is False:
        return "folded"
    return "unknown"


class SessionAnnotator:
    """Load / save / prefill annotations of one recorded session."""

    def __init__(self, session_path) -> None:
        self.data = load_session(session_path)
        self.path = self.data.path
        self.samples = [s for s in self.data.samples if s.get("files", {}).get("frame")]
        self.cal = self.data.calibrations[self.data.initial_checksum]
        (self.path / "annotations").mkdir(exist_ok=True)

    @property
    def ids(self) -> List[str]:
        return [s["id"] for s in self.samples]

    def ann_path(self, sid: str) -> Path:
        return self.path / "annotations" / f"{sid}.json"

    def frame_path(self, sid: str) -> Path:
        s = next(x for x in self.samples if x["id"] == sid)
        return self.path / s["files"]["frame"]

    def load(self, sid: str) -> Optional[dict]:
        p = self.ann_path(sid)
        return json.loads(p.read_text()) if p.exists() else None

    def status(self, sid: str) -> str:
        d = self.load(sid)
        return "unreviewed" if d is None else d.get("status", "complete")

    def progress(self) -> Dict[str, int]:
        counts = {s: 0 for s in STATUSES}
        for sid in self.ids:
            counts[self.status(sid)] += 1
        counts["total"] = len(self.ids)
        counts["annotated"] = counts["partial"] + counts["complete"]
        return counts

    def next_unreviewed(self, after: Optional[str] = None) -> Optional[str]:
        ids = self.ids
        start = ids.index(after) + 1 if after in ids else 0
        for sid in ids[start:] + ids[:start]:
            if self.status(sid) == "unreviewed":
                return sid
        return None

    def observer_readings(self, sid: str) -> Dict[str, dict]:
        s = next(x for x in self.samples if x["id"] == sid)
        out = {}
        for k in ("observation", "tracked_state", "diagnostics"):
            f = s.get("files", {}).get(k)
            if f and (self.path / f).exists():
                out[k] = json.loads((self.path / f).read_text())
        return out

    def prefill(self, sid: str) -> dict:
        """A draft from the observer's fused state (status stays unreviewed)."""
        st = self.observer_readings(sid).get("tracked_state")
        n, hero = self.cal.num_seats, self.cal.hero_seat
        d = self.empty(sid)
        d["prefilled_from_observer"] = True
        if not st:
            return d
        snap = st["snapshot"]
        hc = [c for c in snap["hero_cards"] if c]
        d.update({"hero_cards": hc if len(hc) == 2 else None, "board": list(snap["board"]),
                  "dealer_seat": snap["dealer"], "actor": snap["actor"], "pot": snap["pot"],
                  "pot_total": st.get("pot_total")})
        seats = []
        for i in range(n):
            if not snap["occupied"][i]:
                seats.append({"seat": i, "occupied": False})
                continue
            seats.append({"seat": i, "occupied": True, "stack": snap["stacks"][i],
                          "bet": snap["bets"][i], "active": snap["in_hand"][i],
                          "all_in": snap["all_in"][i]})
        d["seats"] = seats
        d["hero_seat"] = hero
        return d

    def empty(self, sid: str) -> dict:
        return {"format": ANNOTATION_FORMAT, "image": f"frames/{sid}.png",
                "role": "unassigned", "status": "unreviewed",
                "num_seats": self.cal.num_seats, "hero_seat": self.cal.hero_seat,
                "session": self.data.session.get("id"), "sample": sid}

    def save(self, sid: str, form: dict) -> dict:
        """Write an annotation built by :func:`build_annotation`; raises
        ``ValueError`` with the problems if it is inconsistent."""
        problems = validate_annotation(form)
        if problems:
            raise ValueError("; ".join(problems))
        self.ann_path(sid).write_text(json.dumps(form, indent=1))
        return form


def build_annotation(base: dict, *, status: str, role: str = "unassigned",
                     hero_cards: Sequence[str] = ("", ""), board: str = "",
                     street: str = "unknown", dealer: str = "unknown",
                     actor: str = "unknown", pot: str = "", pot_total: str = "",
                     seats: Optional[List[dict]] = None, notes: str = "") -> dict:
    """Turn form inputs (strings; blank = unknown) into a partial annotation.

    ``seats``: one dict per seat with ``status`` (a SEAT_STATUSES entry),
    ``stack`` and ``bet`` strings. ``dealer`` / ``actor``: "unknown",
    "none" or a seat number as string.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    d = {k: v for k, v in base.items() if k in (
        "format", "image", "num_seats", "hero_seat", "session", "sample")}
    d.update({"status": status, "role": role})
    known: List[str] = []
    h = [parse_card(c) for c in hero_cards]
    if all(c is not None for c in h):
        d["hero_cards"] = h
        known.append("hero_cards")
    elif any(c is not None for c in h):
        raise ValueError("give both hero cards or neither")
    b = parse_cards(board)
    if b is not None:
        d["board"] = b
        known.append("board")
    if street != "unknown":
        d["street"] = street
        known.append("street")
        if b is None and STREETS[street] == 0:
            d["board"] = []
            known.append("board")
    for key, val in (("dealer_seat", dealer), ("actor", actor)):
        if val == "unknown":
            continue
        d[key] = None if val == "none" else int(val)
        known.append(key)
    for key, val in (("pot", pot), ("pot_total", pot_total)):
        v = parse_amount(val)
        if v is not None:
            d[key] = v
            known.append(key)
    out_seats = []
    for i, s in enumerate(seats or []):
        entry = {"seat": i, **seat_status_to_fields(s.get("status", "unknown"))}
        if entry.get("occupied") is not False:
            for k in ("stack", "bet"):
                v = parse_amount(s.get(k, ""))
                if v is not None:
                    entry[k] = v
            if entry.get("all_in") and "stack" not in entry:
                entry["stack"] = 0.0
        if len(entry) > 1:
            out_seats.append(entry)
    if out_seats:
        d["seats"] = out_seats
        known.append("seats")
    d["annotated"] = [g for g in GROUPS if g in set(known)]
    if notes.strip():
        d["notes"] = notes.strip()
    return d
