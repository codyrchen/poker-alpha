"""Local session storage in a transparent SQLite file.

Everything is plain SQL tables with JSON text columns (inspectable with any
SQLite browser). Schema version is recorded in ``meta``; opening a database
with a different version raises rather than guessing.

Tables: ``sessions``, ``hands`` (canonical events, result, statistics
summary), ``snapshots`` (hero-view states), ``decisions`` (state, full
recommendation, the hero's actual action and EV comparison).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from ..decision import DecisionConfig, recommend_action
from ..decision.report import DecisionReport
from ..history import Event, replay_hand
from ..history.replay import ReplayResult
from ..holdem.adapters import observed_to_dict
from .serialize import report_to_dict, summary_to_dict

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, started REAL NOT NULL, source TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS hands (
    id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
    hand_key TEXT NOT NULL, hand_index INTEGER NOT NULL,
    events_json TEXT NOT NULL, result_json TEXT NOT NULL,
    summary_json TEXT NOT NULL, big_blind REAL NOT NULL,
    hero_seat INTEGER, hero_net REAL);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY, hand_id INTEGER NOT NULL REFERENCES hands(id),
    seq INTEGER NOT NULL, state_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, hand_id INTEGER NOT NULL REFERENCES hands(id),
    seq INTEGER NOT NULL, street TEXT NOT NULL, state_json TEXT NOT NULL,
    report_json TEXT, actual_kind TEXT NOT NULL, actual_amount REAL,
    actual_label TEXT, recommended TEXT, ev_actual REAL, ev_best REAL,
    ev_loss REAL, confidence TEXT);
"""


def actual_label(kind: str, amount: Optional[float],
                 report: Optional[DecisionReport]) -> Optional[str]:
    """Map the hero's real action onto the report's candidate labels."""
    if kind in ("fold", "check", "call"):
        return kind
    if report is None:
        return None
    if kind == "all_in":
        labels = [c.label for c in report.candidates]
        if "all_in" in labels:
            return "all_in"
    best, gap = None, float("inf")
    for c in report.candidates:
        if c.kind in ("bet", "raise", "all_in") and amount is not None:
            g = abs(c.amount_to - amount)
            if g < gap:
                best, gap = c.label, g
    return best


class SessionStore:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(_SCHEMA)
        row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES ('schema_version', ?)",
                            (str(SCHEMA_VERSION),))
            self.db.commit()
        elif int(row[0]) != SCHEMA_VERSION:
            raise ValueError(f"session database schema {row[0]} != {SCHEMA_VERSION}")

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def create_session(self, source: str, notes: str = "",
                       started: Optional[float] = None) -> int:
        cur = self.db.execute(
            "INSERT INTO sessions (started, source, notes) VALUES (?, ?, ?)",
            (time.time() if started is None else started, source, notes))
        self.db.commit()
        return int(cur.lastrowid)

    def add_hand(self, session_id: int, events: Sequence[Event],
                 result: ReplayResult,
                 reports: Sequence[Optional[DecisionReport]] = ()) -> int:
        hero = result.started.hero_seat
        res = {"awards": {str(k): v for k, v in result.awards.items()},
               "net": {str(k): v for k, v in result.net.items()},
               "warnings": list(result.warnings)}
        cur = self.db.execute(
            "INSERT INTO hands (session_id, hand_key, hand_index, events_json,"
            " result_json, summary_json, big_blind, hero_seat, hero_net)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, result.hand_id, result.summary.hand_index,
             json.dumps([e.to_dict() for e in events]), json.dumps(res),
             json.dumps(summary_to_dict(result.summary)),
             result.started.big_blind, hero,
             None if hero is None else result.net.get(hero)))
        hand_id = int(cur.lastrowid)
        for seq, (_, obs) in enumerate(result.snapshots):
            self.db.execute("INSERT INTO snapshots (hand_id, seq, state_json)"
                            " VALUES (?, ?, ?)",
                            (hand_id, seq, json.dumps(observed_to_dict(obs))))
        for seq, dec in enumerate(result.decisions):
            rep = reports[seq] if seq < len(reports) else None
            label = actual_label(dec.action, dec.amount, rep)
            ev_actual = ev_best = loss = None
            if rep is not None:
                evs = {c.label: c.ev_bb for c in rep.candidates if c.ev_bb is not None}
                if evs:
                    ev_best = max(evs.values())
                    if label in evs:
                        ev_actual = evs[label]
                        loss = ev_best - ev_actual
            self.db.execute(
                "INSERT INTO decisions (hand_id, seq, street, state_json,"
                " report_json, actual_kind, actual_amount, actual_label,"
                " recommended, ev_actual, ev_best, ev_loss, confidence)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (hand_id, seq, dec.state.street.name,
                 json.dumps(observed_to_dict(dec.state)),
                 None if rep is None else json.dumps(report_to_dict(rep)),
                 dec.action, dec.amount, label,
                 None if rep is None else rep.recommended, ev_actual, ev_best,
                 loss, None if rep is None else rep.confidence))
        self.db.commit()
        return hand_id

    # -- queries -------------------------------------------------------------

    def sessions(self) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT id, started, source, notes FROM sessions"
                               " ORDER BY id").fetchall()
        return [dict(zip(("id", "started", "source", "notes"), r)) for r in rows]

    def hands(self, session_id: int) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, hand_key, hand_index, result_json, summary_json,"
            " big_blind, hero_seat, hero_net FROM hands WHERE session_id=?"
            " ORDER BY id", (session_id,)).fetchall()
        keys = ("id", "hand_key", "hand_index", "result", "summary",
                "big_blind", "hero_seat", "hero_net")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["result"] = json.loads(d["result"])
            d["summary"] = json.loads(d["summary"])
            out.append(d)
        return out

    def decisions(self, session_id: int) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT d.id, h.hand_key, d.seq, d.street, d.report_json,"
            " d.actual_kind, d.actual_amount, d.actual_label, d.recommended,"
            " d.ev_actual, d.ev_best, d.ev_loss, d.confidence, h.id"
            " FROM decisions d JOIN hands h ON d.hand_id = h.id"
            " WHERE h.session_id=? ORDER BY h.id, d.seq", (session_id,)).fetchall()
        keys = ("id", "hand_key", "seq", "street", "report", "actual_kind",
                "actual_amount", "actual_label", "recommended", "ev_actual",
                "ev_best", "ev_loss", "confidence", "hand_id")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["report"] = None if d["report"] is None else json.loads(d["report"])
            out.append(d)
        return out


def import_hands(store: SessionStore, hands: Sequence[Sequence[Event]],
                 source: str = "hand history",
                 config: Optional[DecisionConfig] = None,
                 analyze: bool = True) -> int:
    """Replay, analyse (optionally) and store a list of hands as one session."""
    sid = store.create_session(source)
    for k, events in enumerate(hands):
        res = replay_hand(events, hand_index=k)
        reports = []
        if analyze:
            for dec in res.decisions:
                reports.append(recommend_action(dec.state, config=config))
        store.add_hand(sid, events, res, reports)
    return sid
