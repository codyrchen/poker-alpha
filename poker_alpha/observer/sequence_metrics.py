"""Sequence (event) metrics for annotated observer sessions (Phase 46).

Ground truth is sparse: a human annotates some kept frames. Between two
consecutive annotated frames the *net* change of the annotated state defines
the true events of that interval:

* ``new_hand`` — hero cards changed, the dealer moved or the board shrank;
* ``street``   — the board grew (tracker event ``board``);
* ``bet``      — a seat's bet grew (per seat);
* ``fold``     — a seat went from in-hand to out (per seat).

The observed events are the tracker events of every frame in that interval,
replayed from the session's raw-reading stream. Matching is by (kind, seat)
sets per interval, so several observed bets by one seat in one interval
count once. An interval that contains a true new hand is scored on new-hand
detection only (bets / folds across hands are not comparable by diffing).

Whole-session diagnostics (no ground truth needed): state flicker (the fused
state reverts to the previous value within 3 frames), card persistence
violations (hero cards or board cards change within one tracked hand without
the board just growing) and stack persistence violations (a stack grows
within one tracked hand while the pot did not drop).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .annotations import Annotation, load_fixture_dir
from .session_replay import load_session, replay_stream

EVENT_KINDS = ("new_hand", "street", "bet", "fold")
_OBS_KIND = {"new_hand": "new_hand", "board": "street", "bet": "bet", "fold": "fold"}


def truth_events(a: Annotation, b: Annotation) -> set:
    ev = set()
    if a.knows("hero_cards") and b.knows("hero_cards") and a.hero_cards and b.hero_cards \
            and tuple(a.hero_cards) != tuple(b.hero_cards):
        ev.add(("new_hand", None))
    if a.knows("dealer_seat") and b.knows("dealer_seat") and a.dealer_seat is not None \
            and b.dealer_seat is not None and a.dealer_seat != b.dealer_seat:
        ev.add(("new_hand", None))
    if a.board is not None and b.board is not None:
        if len(b.board) < len(a.board):
            ev.add(("new_hand", None))
        elif len(b.board) > len(a.board):
            ev.add(("street", None))
    if ("new_hand", None) in ev:
        return {("new_hand", None)}
    if a.knows("seats") and b.knows("seats"):
        for s in range(min(a.num_seats, b.num_seats)):
            x, y = a.seats.get(s), b.seats.get(s)
            if x is None or y is None:
                continue
            if x.bet is not None and y.bet is not None and y.bet > x.bet + 1e-9:
                ev.add(("bet", s))
            if x.active is True and y.active is False and x.occupied and y.occupied:
                ev.add(("fold", s))
    return ev


def _observed(events: Sequence[tuple]) -> set:
    out = set()
    for _, kind, seat, _amount in events:
        k = _OBS_KIND.get(kind)
        if k is not None:
            out.add((k, seat if k in ("bet", "fold") else None))
    return out


def _pr(tp: int, fp: int, fn: int) -> dict:
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def flicker_count(states: Sequence[tuple], window: int = 3) -> int:
    """A -> B -> ... -> A within ``window`` frames after a change."""
    n = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            if states[i - 1] in states[i + 1:i + 1 + window]:
                n += 1
    return n


def _key(state: dict) -> tuple:
    return tuple((k, repr(v)) for k, v in sorted(state.items()))


def persistence_violations(steps) -> Dict[str, int]:
    cards = stacks = 0
    prev = None
    for st in steps:
        if prev is not None and st.hand_number == prev.hand_number:
            ps, cs = prev.state, st.state
            if all(ps["hero_cards"]) and all(cs["hero_cards"]) and \
                    list(ps["hero_cards"]) != list(cs["hero_cards"]):
                cards += 1
            pb, cb = list(ps["board"]), list(cs["board"])
            if pb and cb[:len(pb)] != pb:
                cards += 1
            pot_dropped = (ps["pot"] is not None and cs["pot"] is not None
                           and cs["pot"] < ps["pot"] - 1e-9)
            for a, b in zip(ps["stacks"], cs["stacks"]):
                if a is not None and b is not None and b > a + 1e-9 and not pot_dropped:
                    stacks += 1
        prev = st
    return {"card_persistence_violations": cards, "stack_persistence_violations": stacks}


def sequence_metrics(session_path, annotations: Optional[List[Annotation]] = None,
                     roles: Optional[Sequence[str]] = None) -> dict:
    """Event precision / recall between annotated frames plus whole-session
    stability diagnostics. ``roles`` restricts the annotated frames used."""
    data = load_session(session_path)
    steps = replay_stream(data)
    ordered = [steps[k] for k in sorted(steps)]
    anns = annotations if annotations is not None else load_fixture_dir(data.path)
    by_frame: Dict[int, Annotation] = {}
    for a in anns:
        if roles is not None and a.role not in roles:
            continue
        try:
            by_frame[int(a.name)] = a
        except ValueError:
            continue
    frames = sorted(by_frame)
    tallies = {k: [0, 0, 0] for k in EVENT_KINDS}
    intervals = []
    for f0, f1 in zip(frames, frames[1:]):
        truth = truth_events(by_frame[f0], by_frame[f1])
        obs_events = [e for n in sorted(steps) if f0 < n <= f1 for e in steps[n].events]
        obs = _observed(obs_events)
        if ("new_hand", None) in truth:
            obs = {e for e in obs if e[0] == "new_hand"}
        for k in EVENT_KINDS:
            t = {e for e in truth if e[0] == k}
            o = {e for e in obs if e[0] == k}
            tallies[k][0] += len(t & o)
            tallies[k][1] += len(o - t)
            tallies[k][2] += len(t - o)
        intervals.append({"from": f0, "to": f1, "truth": sorted(map(list, truth), key=str),
                          "observed": sorted(map(list, obs), key=str)})
    fp_total = sum(v[1] for v in tallies.values())
    stream_frames = len(ordered)
    duration_min = None
    if data.samples:
        duration_min = max(s.get("monotonic", 0.0) for s in data.samples) / 60.0
    elapsed = data.session.get("elapsed_s")
    if elapsed:
        duration_min = elapsed / 60.0
    out = {
        "annotated_frames": len(frames), "intervals": len(intervals),
        "events": {k: _pr(*v) for k, v in tallies.items()},
        "false_events": fp_total,
        "false_events_per_frame": fp_total / stream_frames if stream_frames else None,
        "false_events_per_minute": (fp_total / duration_min) if duration_min else None,
        "streamed_frames": stream_frames,
        "state_flicker": flicker_count([_key(s.state) for s in ordered]),
        **persistence_violations(ordered),
        "interval_detail": intervals,
        "note": "events matched as (kind, seat) sets per interval between consecutive "
                "annotated frames; intervals with a true new hand score new-hand only",
    }
    if len(frames) < 2:
        out["status"] = "NOT MEASURED: fewer than 2 annotated frames"
    return out
