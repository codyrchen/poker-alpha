"""Deterministic replay of an observer test session (Phase 42).

    python -m poker_alpha.observer.session_replay ~/pokeralpha_sessions/<id>
        [--mode stored|recompute] [--calibration FILE] [--compare-calibration FILE]
        [--blinds SB BB] [--no-decisions] [--write-report OUT.json]
        [--compare-report EARLIER.json]

Modes
-----
``stored``
    Replays the raw readings of *every* processed frame
    (``observations/stream.jsonl``) through a fresh :class:`StateTracker`,
    honouring every tracker reset recorded in the stream, and checks that
    the fused state at each retained sample and the tracker events equal what
    the live session recorded. A mismatch means the tracker is not
    deterministic or changed since the recording.
``recompute``
    Re-runs recognition on the retained frame images (with the session's
    calibration, or ``--calibration``) and compares the readings field by
    field with the stored ones, then compares tracker states / events /
    decisions over the retained frames (stored readings vs recomputed
    readings, same frames, same order). With ``--compare-calibration`` the
    baseline is a recompute with the session calibration and the candidate a
    recompute with the given one.

Only retained frames have images, so recompute-mode tracker states describe
the retained subsequence, not the live sequence; differences are reported
between two replays of the same subsequence (apples to apples).

To compare two code versions without Git gymnastics: run with
``--write-report a.json`` on one checkout and with ``--compare-report
a.json`` on the other.

Read-only and local: reads the session folder, optionally writes the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .calibration import TableCalibration
from .fusion import StateTracker
from .pokernow import FieldReading, FrameObservation, PokerNowStyleAdapter
from .session import calibration_checksum, snapshot_dict

REPORT_FORMAT = "pokeralpha.session_replay_report/v1"
TRACKER_EVENT_KINDS = {"bet", "stack_decrease", "stack_increase", "bets_collected",
                       "fold", "board", "new_hand"}


@dataclass
class SessionData:
    path: Path
    session: dict
    calibrations: Dict[str, TableCalibration]   # checksum -> calibration
    initial_checksum: str
    samples: List[dict]
    stream: List[dict]
    events: List[dict]

    def calibration(self, checksum: Optional[str]) -> TableCalibration:
        if checksum in self.calibrations:
            return self.calibrations[checksum]
        return self.calibrations[self.initial_checksum]

    def blinds(self) -> Tuple[float, float]:
        for row in self.stream:
            if row.get("reset") and row.get("bb"):
                return float(row["sb"]), float(row["bb"])
        b = (self.session.get("meta") or {}).get("blinds")
        if b and b[1]:
            return float(b[0]), float(b[1])
        return 0.5, 1.0


def _jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # a truncated last line after a crash
                continue
    return out


def load_session(path) -> SessionData:
    path = Path(path).expanduser()
    if not (path / "session.json").exists():
        raise FileNotFoundError(f"{path} is not an observer session (no session.json)")
    session = json.loads((path / "session.json").read_text())
    initial = TableCalibration.load(path / "calibration.json")
    initial_sum = calibration_checksum(initial)
    cals = {initial_sum: initial}
    for p in sorted((path / "calibrations").glob("*.json")):
        c = TableCalibration.load(p)
        cals[calibration_checksum(c)] = c
    if (path / "manifest.json").exists():
        samples = json.loads((path / "manifest.json").read_text())["samples"]
        seen = {s["id"] for s in samples}
        # samples streamed after the last manifest write (e.g. a crash)
        samples += [s for s in _jsonl(path / "manifest.jsonl") if s["id"] not in seen]
    else:
        samples = _jsonl(path / "manifest.jsonl")
    dedup: Dict[str, dict] = {}
    for s in samples:
        dedup[s["id"]] = s                   # the last manifest line of a sample wins
    samples = sorted(dedup.values(), key=lambda s: s["frame_number"])
    return SessionData(path, session, cals, initial_sum, samples,
                       _jsonl(path / "observations" / "stream.jsonl"),
                       _jsonl(path / "events" / "events.jsonl"))


def observation_from_stream(row: dict) -> FrameObservation:
    obs = FrameObservation(row.get("t"), tuple(row["bbox"]))
    for k, (value, conf, region) in row["f"].items():
        obs.fields[k] = FieldReading(value, float(conf), region, row.get("t"))
    return obs


def observation_from_file(d: dict, timestamp=None) -> FrameObservation:
    obs = FrameObservation(timestamp, tuple(d["table_bbox"]))
    for k, f in d["fields"].items():
        obs.fields[k] = FieldReading(f["value"], float(f["confidence"]), f["region"], timestamp)
    return obs


def _event_tuple(e, n) -> tuple:
    amount = None if e.get("amount") is None else round(float(e["amount"]), 6)
    return (n, e["kind"], e.get("seat"), amount)


@dataclass
class TrackStep:
    n: int
    state: dict
    critical_confidence: float
    hand_number: int
    events: List[tuple] = field(default_factory=list)


def _track_state(tracker: StateTracker) -> dict:
    return snapshot_dict(tracker.snapshot())


def replay_stream(data: SessionData, calibration: Optional[TableCalibration] = None,
                  blinds: Optional[Tuple[float, float]] = None) -> Dict[int, TrackStep]:
    """Feed the stored raw readings of every frame through fresh trackers."""
    sb, bb = blinds or data.blinds()
    tracker: Optional[StateTracker] = None
    out: Dict[int, TrackStep] = {}
    for row in data.stream:
        if row.get("reset"):
            cal = calibration or data.calibration(row.get("calibration"))
            tracker = StateTracker(cal, row.get("sb") or sb, row.get("bb") or bb)
            continue
        if not row.get("ok") or tracker is None:
            continue
        events = tracker.update(observation_from_stream(row))
        n = int(row["n"])
        out[n] = TrackStep(n, _track_state(tracker), tracker.critical_confidence(),
                           tracker.hand_number,
                           [(n, e.kind, e.seat, None if e.amount is None
                             else round(float(e.amount), 6)) for e in events])
    return out


def _sample_obs(data: SessionData, sample: dict) -> Optional[dict]:
    f = sample.get("files", {}).get("observation")
    if not f or not (data.path / f).exists():
        return None
    return json.loads((data.path / f).read_text())


def _sample_state(data: SessionData, sample: dict) -> Optional[dict]:
    f = sample.get("files", {}).get("tracked_state")
    if not f or not (data.path / f).exists():
        return None
    return json.loads((data.path / f).read_text())


def _norm(v):
    return json.loads(json.dumps(v))      # tuples -> lists, as stored


def compare_states(a: dict, b: dict) -> List[dict]:
    diffs = []
    for k in sorted(set(a) | set(b)):
        if _norm(a.get(k)) != _norm(b.get(k)):
            diffs.append({"field": k, "baseline": _norm(a.get(k)), "candidate": _norm(b.get(k))})
    return diffs


def compare_fields(base: FrameObservation, cand: FrameObservation) -> List[dict]:
    diffs = []
    for k in sorted(set(base.fields) | set(cand.fields)):
        a, b = base.fields.get(k), cand.fields.get(k)
        va, vb = (a.value if a else None), (b.value if b else None)
        ca, cb = (a.confidence if a else 0.0), (b.confidence if b else 0.0)
        if _norm(va) != _norm(vb) or abs(ca - cb) > 1e-9:
            diffs.append({"field": k, "baseline": _norm(va), "candidate": _norm(vb),
                          "value_changed": _norm(va) != _norm(vb),
                          "confidence_delta": round(cb - ca, 6)})
    return diffs


def _decide(tracker: StateTracker, min_conf: float, sims: int) -> dict:
    from ..decision import DecisionConfig, recommend_action
    from .live import critical_check

    check = critical_check(tracker, min_conf)
    if not check.ok:
        return {"decision": None, "blocked": list(check.problems)}
    try:
        rep = recommend_action(tracker.to_observed_state(),
                               config=DecisionConfig(equity_simulations=sims, seed=0,
                                                     observer_confidence=check.critical_confidence))
    except Exception as exc:  # noqa: BLE001 - recorded
        return {"decision": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"decision": rep.recommended, "method": rep.method,
            "sources": sorted({c.source for c in rep.candidates})}


def _track_samples(data: SessionData, obs_list, calibration: Optional[TableCalibration],
                   blinds, decisions: bool, min_conf: float, sims: int):
    """Tracker over the retained subsequence; resets follow the stream."""
    sb, bb = blinds or data.blinds()
    resets = [int(r["n"]) for r in data.stream if r.get("reset")]
    cal_at = {int(r["n"]): r.get("calibration") for r in data.stream if r.get("reset")}
    tracker, last_reset = None, None
    out = {}
    for sample, obs in obs_list:
        n = sample["frame_number"]
        prior = [r for r in resets if r <= n]
        r = prior[-1] if prior else None
        if tracker is None or r != last_reset:
            cal = calibration or data.calibration(cal_at.get(r))
            tracker, last_reset = StateTracker(cal, sb, bb), r
        if obs is None:
            continue
        events = tracker.update(obs)
        rec = {"state": _track_state(tracker),
               "critical_confidence": tracker.critical_confidence(),
               "hand_number": tracker.hand_number,
               "events": [(n, e.kind, e.seat, None if e.amount is None
                           else round(float(e.amount), 6)) for e in events]}
        if decisions:
            rec.update(_decide(tracker, min_conf, sims))
        out[sample["id"]] = rec
    return out


def replay(path, mode: str = "stored", calibration: Optional[TableCalibration] = None,
           compare_calibration: Optional[TableCalibration] = None,
           blinds: Optional[Tuple[float, float]] = None, decisions: bool = True,
           min_conf: float = 0.5, sims: int = 400) -> dict:
    data = load_session(path)
    report = {"format": REPORT_FORMAT, "session": data.session.get("id"),
              "session_path": str(data.path), "mode": mode,
              "samples": len(data.samples), "stream_frames": sum(1 for r in data.stream
                                                                 if r.get("ok")),
              "blinds": list(blinds or data.blinds())}
    if mode == "stored":
        steps = replay_stream(data, calibration, blinds)
        per = []
        mismatched = 0
        for s in data.samples:
            stored = _sample_state(data, s)
            step = steps.get(s["frame_number"])
            if stored is None or step is None:
                per.append({"sample": s["id"], "status": "missing",
                            "stored_state": stored is not None, "replayed": step is not None})
                continue
            diffs = compare_states(stored["snapshot"], step.state)
            if stored.get("hand_number") != step.hand_number:
                diffs.append({"field": "hand_number", "baseline": stored.get("hand_number"),
                              "candidate": step.hand_number})
            if abs(stored["critical_confidence"] - step.critical_confidence) > 1e-9:
                diffs.append({"field": "critical_confidence",
                              "baseline": stored["critical_confidence"],
                              "candidate": step.critical_confidence})
            mismatched += bool(diffs)
            per.append({"sample": s["id"], "status": "match" if not diffs else "differs",
                        "differences": diffs})
        live_events = [_event_tuple(e, e.get("frame_number")) for e in data.events
                       if e["kind"] in TRACKER_EVENT_KINDS and e.get("frame_number") is not None]
        replay_events = [e for st in sorted(steps.values(), key=lambda x: x.n) for e in st.events]
        report.update({
            "per_sample": per, "samples_differing": mismatched,
            "events_live": len(live_events), "events_replayed": len(replay_events),
            "events_only_live": [list(e) for e in live_events if e not in replay_events],
            "events_only_replay": [list(e) for e in replay_events if e not in live_events],
        })
        report["reproduced"] = (mismatched == 0 and not report["events_only_live"]
                                and not report["events_only_replay"]
                                and all(p["status"] == "match" for p in per))
        return report

    if mode != "recompute":
        raise ValueError("mode must be 'stored' or 'recompute'")
    from PIL import Image

    base_obs, cand_obs, per = [], [], []
    for s in data.samples:
        frame = s.get("files", {}).get("frame")
        stored = _sample_obs(data, s)
        diag_cal = None
        st = _sample_state(data, s)
        if st is not None:
            diag_cal = st.get("calibration_checksum")
        cal_base = calibration or data.calibration(diag_cal)
        if frame is None or not (data.path / frame).exists():
            per.append({"sample": s["id"], "status": "no_frame"})
            base_obs.append((s, None))
            cand_obs.append((s, None))
            continue
        img = Image.open(data.path / frame).convert("RGB")
        if compare_calibration is not None:
            base = PokerNowStyleAdapter(cal_base).read_frame(img)
            cand_cal = compare_calibration
        else:
            base = observation_from_file(stored) if stored else None
            cand_cal = cal_base
        try:
            cand = PokerNowStyleAdapter(cand_cal).read_frame(img)
        except Exception as exc:  # noqa: BLE001
            per.append({"sample": s["id"], "status": "recognition_failed",
                        "error": f"{type(exc).__name__}: {exc}"})
            base_obs.append((s, base))
            cand_obs.append((s, None))
            continue
        diffs = compare_fields(base, cand) if base is not None else []
        per.append({"sample": s["id"], "status": "match" if not diffs else "differs",
                    "field_differences": diffs,
                    "values_changed": sum(d["value_changed"] for d in diffs)})
        base_obs.append((s, base))
        cand_obs.append((s, cand))
    tb = _track_samples(data, base_obs, calibration, blinds, decisions, min_conf, sims)
    tc = _track_samples(data, cand_obs, compare_calibration or calibration, blinds,
                        decisions, min_conf, sims)
    tracker_diffs, event_diffs, decision_diffs = [], [], []
    for s in data.samples:
        a, b = tb.get(s["id"]), tc.get(s["id"])
        if a is None or b is None:
            continue
        d = compare_states(a["state"], b["state"])
        if d:
            tracker_diffs.append({"sample": s["id"], "differences": d})
        if a["events"] != b["events"]:
            event_diffs.append({"sample": s["id"], "baseline": a["events"],
                                "candidate": b["events"]})
        if decisions and (a.get("decision"), a.get("method")) != (b.get("decision"),
                                                                  b.get("method")):
            decision_diffs.append({"sample": s["id"],
                                   "baseline": {k: a.get(k) for k in ("decision", "method", "blocked")},
                                   "candidate": {k: b.get(k) for k in ("decision", "method", "blocked")}})
    report.update({
        "baseline": "recompute with session calibration" if compare_calibration is not None
        else "stored readings",
        "candidate": "recompute with --compare-calibration" if compare_calibration is not None
        else "recompute with " + ("--calibration" if calibration else "session calibration"),
        "per_sample": per,
        "samples_with_value_changes": sum(1 for p in per if p.get("values_changed")),
        "tracker_differences": tracker_diffs, "event_differences": event_diffs,
        "decision_differences": decision_diffs,
        "candidate_results": {k: {kk: v for kk, v in r.items() if kk != "events"}
                              for k, r in tc.items()},
        "note": "recompute replays retained frames only (the subsequence that has images)",
    })
    return report


def compare_reports(old: dict, new: dict) -> dict:
    """Per-sample differences between two replay reports (e.g. two commits)."""
    diffs = []
    o = old.get("candidate_results", {})
    n = new.get("candidate_results", {})
    for k in sorted(set(o) | set(n)):
        a, b = o.get(k), n.get(k)
        if a is None or b is None:
            diffs.append({"sample": k, "only_in": "old" if b is None else "new"})
            continue
        d = compare_states(a.get("state", {}), b.get("state", {}))
        if (a.get("decision"), a.get("method")) != (b.get("decision"), b.get("method")):
            d.append({"field": "decision", "baseline": [a.get("decision"), a.get("method")],
                      "candidate": [b.get("decision"), b.get("method")]})
        if abs((a.get("critical_confidence") or 0) - (b.get("critical_confidence") or 0)) > 1e-9:
            d.append({"field": "critical_confidence", "baseline": a.get("critical_confidence"),
                      "candidate": b.get("critical_confidence")})
        if d:
            diffs.append({"sample": k, "differences": d})
    return {"samples_compared": len(set(o) & set(n)), "samples_differing": len(diffs),
            "differences": diffs}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--mode", choices=("stored", "recompute"), default="stored")
    p.add_argument("--calibration", type=Path, help="recompute with this calibration")
    p.add_argument("--compare-calibration", type=Path,
                   help="recompute with the session calibration AND this one, compare")
    p.add_argument("--blinds", nargs=2, type=float, metavar=("SB", "BB"))
    p.add_argument("--no-decisions", action="store_true")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--equity-sims", type=int, default=400)
    p.add_argument("--write-report", type=Path)
    p.add_argument("--compare-report", type=Path,
                   help="report written earlier (e.g. by another commit) to diff against")
    a = p.parse_args(argv)
    mode = "recompute" if a.compare_calibration else a.mode
    rep = replay(a.session, mode,
                 TableCalibration.load(a.calibration) if a.calibration else None,
                 TableCalibration.load(a.compare_calibration) if a.compare_calibration else None,
                 tuple(a.blinds) if a.blinds else None, not a.no_decisions,
                 a.min_confidence, a.equity_sims)
    if a.compare_report:
        rep["compare_report"] = compare_reports(json.loads(a.compare_report.read_text()), rep)
    print(f"session {rep['session']} · mode {rep['mode']} · {rep['samples']} retained samples · "
          f"{rep['stream_frames']} streamed frames")
    if mode == "stored":
        print(f"reproduced: {rep['reproduced']} · samples differing {rep['samples_differing']} · "
              f"events live {rep['events_live']} / replayed {rep['events_replayed']}")
    else:
        print(f"baseline: {rep['baseline']} · candidate: {rep['candidate']}")
        print(f"samples with changed values {rep['samples_with_value_changes']} · "
              f"tracker diffs {len(rep['tracker_differences'])} · event diffs "
              f"{len(rep['event_differences'])} · decision diffs {len(rep['decision_differences'])}")
    if "compare_report" in rep:
        c = rep["compare_report"]
        print(f"vs earlier report: {c['samples_differing']} of {c['samples_compared']} samples differ")
    if a.write_report:
        a.write_report.write_text(json.dumps(rep, indent=1, default=str))
        print(f"wrote {a.write_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
