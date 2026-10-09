"""Per-region observer diagnostics (Phase 47). Read-only: explains what the
recognizers read and what the tracker did with it; never changes either.

:func:`capture_tracker_state` snapshots every field tracker before an
update (``LiveObserverSession`` does this each step); :func:`region_rows`
then explains, per recognition region, the raw reading, its confidence, the
fused value and why the tracker accepted or did not accept it.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

GROUPS = ("Hero", "Board", "Seats", "Pot/Bets", "Dealer/Actor")


def capture_tracker_state(tracker) -> Dict[str, tuple]:
    if tracker is None:
        return {}
    out = {k: (f.has_value, f.stable, f.stable_confidence, f.candidate,
               f.candidate_count, f.rejected, f.pinned)
           for k, f in tracker.fields.items()}
    out["__flags__"] = tracker.flag_total
    return out


def _flag_for(name: str, flags: List[str]) -> Optional[str]:
    if name.endswith(".stack"):
        seat = name[4:name.index(".")]
        key = f"stack increase on seat {seat}"
        return next((x for x in flags if key in x), None)
    return next((x for x in flags if f"{name} " in x + " " and name.startswith("board")), None)


def region_group(name: str) -> str:
    if name.startswith("hero_card"):
        return "Hero"
    if name.startswith("board"):
        return "Board"
    if name == "pot" or name.endswith("_bet"):
        return "Pot/Bets"
    if name.endswith("_dealer") or name.endswith("_active"):
        return "Dealer/Actor"
    return "Seats"


def _fmt(v):
    return "—" if v is None else v


def field_decision(name: str, reading, before: Optional[tuple], tracker,
                   new_flags: List[str]) -> Dict[str, object]:
    """Why the tracker did / did not take this reading (explanation only)."""
    f = tracker.fields.get(name)
    if f is None:
        return {"decision": "not tracked", "reason": "derived field"}
    b_has, b_stable, _b_conf, _b_cand, _b_count, b_rej, _ = before or (
        False, None, 0.0, None, 0, f.rejected, f.pinned)
    out = {"previous_accepted": b_stable if b_has else None,
           "fused": f.stable if f.has_value else None,
           "fused_confidence": round(f.stable_confidence, 3) if f.has_value else None,
           "agreeing_frames": f.candidate_count if f.candidate is not None else (
               "stable" if f.has_value else 0)}
    flag = _flag_for(name, new_flags)
    if f.pinned:
        out.update(decision="pinned", reason="manual correction in force")
    elif reading is None:
        out.update(decision="no reading", reason="field not read this frame")
    elif f.rejected > b_rej:
        out.update(decision="rejected", reason=f"confidence {reading.confidence:.2f} below "
                                               f"minimum {f.min_confidence:.2f}")
    elif flag is not None:
        out.update(decision="rejected" if "rejected" in flag else "held", reason=flag)
    elif (f.has_value and (not b_has or f.stable != b_stable)):
        fast = reading.confidence >= f.high_confidence and not f.always_confirm
        out.update(decision="accepted",
                   reason="high confidence" if fast else
                   f"confirmed by {f.confirm_frames}+ agreeing frames")
    elif f.has_value and reading.value == f.stable:
        out.update(decision="agrees", reason="same as the fused value")
    elif f.candidate is not None and reading.value == f.candidate:
        need = getattr(f, "confirm_frames", 2)
        out.update(decision="pending",
                   reason=f"candidate seen in {f.candidate_count}/{need} frames")
    else:
        out.update(decision="not accepted", reason="differs from the fused value")
    return out


def region_rows(session) -> List[dict]:
    """One row per recognition region of the last frame."""
    obs, tracker, cal = session.last_observation, session.tracker, session.calibration
    if obs is None or tracker is None or cal is None:
        return []
    before = session.last_tracker_before or {}
    n_new = tracker.flag_total - before.get("__flags__", tracker.flag_total)
    flags_new = list(tracker.flags[-n_new:]) if n_new > 0 else []
    by_region: Dict[str, List[str]] = {}
    for k, r in obs.fields.items():
        by_region.setdefault(r.region, []).append(k)
    rows = []
    for name, reg in cal.regions.items():
        px = reg.to_pixels(obs.table_bbox)
        fields = sorted(by_region.get(name, []))
        row = {"region": name, "group": region_group(name),
               "normalized": [round(v, 4) for v in reg.to_list()], "pixels": list(px),
               "fields": []}
        for fname in fields:
            reading = obs.fields.get(fname)
            d = {"field": fname, "raw": _fmt(reading.value),
                 "confidence": round(reading.confidence, 3)}
            d.update(field_decision(fname, reading, before.get(fname), tracker, flags_new))
            row["fields"].append(d)
        rows.append(row)
    return rows


def crop_png(session, region: str, scale: int = 2) -> Optional[bytes]:
    obs, frame = session.last_observation, session.last_frame
    if obs is None or frame is None or region not in session.calibration.regions:
        return None
    from .regions import crop

    img = crop(frame, session.calibration.regions[region], obs.table_bbox)
    if img.size[0] == 0 or img.size[1] == 0:
        return None
    img = img.resize((img.size[0] * scale, img.size[1] * scale))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tracker_rows(tracker) -> List[dict]:
    out = []
    for k, f in sorted(tracker.fields.items()):
        out.append({"field": k, "fused": _fmt(f.stable if f.has_value else None),
                    "confidence": round(f.stable_confidence, 3),
                    "candidate": _fmt(f.candidate), "agreeing": f.candidate_count,
                    "rejected_total": f.rejected, "pinned": f.pinned})
    return out


def debug_report(session) -> dict:
    tr = session.tracker
    return {"format": "pokeralpha.observer_debug_report/v1",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "frame": session.frames,
            "calibration": session.calibration.name if session.calibration else None,
            "table_bbox": list(session.last_observation.table_bbox)
            if session.last_observation else None,
            "timings_s": session.last_timings, "step_error": session.last_error,
            "step_warning": session.last_warning,
            "regions": region_rows(session),
            "tracker": tracker_rows(tr) if tr is not None else [],
            "tracker_flags": list(tr.flags[-20:]) if tr is not None else [],
            "critical_confidence": tr.critical_confidence() if tr is not None else None,
            "events": [{"kind": e.kind, "seat": e.seat, "amount": e.amount}
                       for e in session.last_events]}


def write_debug_report(session, directory) -> Dict[str, Path]:
    """JSON + self-contained HTML (crops embedded) in a local folder."""
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".gitignore").exists():
        (d / ".gitignore").write_text("*\n")
    rep = debug_report(session)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jp = d / f"debug-{stamp}-f{session.frames}.json"
    k = 1
    while jp.exists():
        jp = d / f"debug-{stamp}-f{session.frames}-{k}.json"
        k += 1
    jp.write_text(json.dumps(rep, indent=1, default=str))
    rows = []
    for r in rep["regions"]:
        png = crop_png(session, r["region"])
        img = (f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}">'
               if png else "")
        fields = "<br>".join(
            f"{f['field']}: <b>{f['raw']}</b> ({f['confidence']}) → {f['decision']}: "
            f"{f['reason']} · fused {f.get('fused')} · previous {f.get('previous_accepted')}"
            for f in r["fields"]) or "—"
        rows.append(f"<tr><td>{r['group']}</td><td>{r['region']}</td><td>{img}</td>"
                    f"<td>{r['pixels']}<br>{r['normalized']}</td><td>{fields}</td></tr>")
    html = ("<!doctype html><meta charset='utf-8'><title>PokerAlpha observer debug</title>"
            "<style>body{font-family:sans-serif}td{border:1px solid #ccc;padding:4px;"
            "vertical-align:top}</style>"
            f"<h1>Observer debug report · frame {rep['frame']}</h1>"
            f"<p>calibration {rep['calibration']} · table {rep['table_bbox']} · critical "
            f"confidence {rep['critical_confidence']} · local file, never uploaded</p>"
            "<table><tr><th>group</th><th>region</th><th>crop</th><th>box</th>"
            "<th>readings</th></tr>" + "".join(rows) + "</table>")
    hp = jp.with_suffix(".html")
    hp.write_text(html)
    return {"json": jp, "html": hp}
