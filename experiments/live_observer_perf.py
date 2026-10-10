"""Live observer performance profile and soak test (Phase 50).

    python experiments/live_observer_perf.py profile [--frames 120]
    python experiments/live_observer_perf.py soak --minutes 30 [--fps 3]

profile
    Times each stage of one observer step on a mocked screen (no monitor):
    capture (fake source returning a prepared frame — real mss capture time is
    not measurable in CI), table detection, region extraction, OCR, card
    recognition, StateTracker, overlay rendering, one Streamlit rerun of the
    Live screen page (AppTest) and decision computation. Stages are timed by
    wrapping functions from outside; production code is unchanged. Reports
    median / p95 / p99 per stage and the share of the frame budget used at
    0.5 / 1 / 2 / 3 FPS. Frames: synthetic 2-seat tables at 1280x800 and the
    PokerNow tuning frame upscaled 2x (Retina-like size; timing only).

soak
    Runs a recording observer session at the given FPS on a synthetic hand
    timeline (street changes, new hands, occasional blank / misread frames,
    capture errors) for the given wall-clock minutes and samples RSS,
    tracemalloc-free Python object counts, tracker list sizes and session
    disk usage every 30 s.

Writes results/validation/live_observer_perf.json (profile) or
results/validation/live_observer_soak.json (soak).
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from poker_alpha.observer import calibration as calmod  # noqa: E402
from poker_alpha.observer import cards as cardmod  # noqa: E402
from poker_alpha.observer import pokernow as pnmod  # noqa: E402
from poker_alpha.observer import text as textmod  # noqa: E402
from poker_alpha.observer.calibration import TableCalibration  # noqa: E402
from poker_alpha.observer.fusion import StateTracker  # noqa: E402
from poker_alpha.observer.live import (CaptureSettings, LiveObserverSession,  # noqa: E402
                                       draw_overlay, live_check)
from poker_alpha.observer.pokernow import default_layout  # noqa: E402
from poker_alpha.observer.synthetic import (SyntheticSeat, SyntheticTable,  # noqa: E402
                                            render_table)

FPS = (0.5, 1.0, 2.0, 3.0)


def pct(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return {"median_ms": round(float(np.median(v)) * 1e3, 2),
            "p95_ms": round(float(np.percentile(v, 95)) * 1e3, 2),
            "p99_ms": round(float(np.percentile(v, 99)) * 1e3, 2), "n": int(v.size)}


class Timers:
    """Accumulate time spent inside wrapped functions, per frame."""

    def __init__(self):
        self.cur = {}

    def wrap(self, owner, attr, key):
        orig = getattr(owner, attr)
        timers = self

        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                timers.cur[key] = timers.cur.get(key, 0.0) + time.perf_counter() - t
        setattr(owner, attr, wrapper)
        return orig

    def take(self):
        c, self.cur = self.cur, {}
        return c


def _table(i):
    boards = [(), ("Qs", "Jh", "4c"), ("Qs", "Jh", "4c", "2d"), ("Qs", "Jh", "4c", "2d", "9s")]
    return SyntheticTable(seats=[SyntheticSeat("h", 99.0 - i % 3), SyntheticSeat("v", 98.0,
                                                                                 bet=float(i % 2))],
                          dealer=(i // 8) % 2, hero_cards=("As", "Kd") if (i // 8) % 2 == 0
                          else ("7c", "7d"), board=boards[(i // 2) % 4], pot=1.5 + i % 4,
                          actor=0)


def profile(frames: int, out: Path) -> dict:
    cal = default_layout(2, 0)
    synth = [render_table(_table(i), cal) for i in range(16)]
    fix = ROOT / "tests" / "fixtures" / "pokernow"
    real = Image.open(fix / "raw" / "hu_preflop_0001.png").convert("RGB")
    real2x = real.resize((real.width * 2, real.height * 2), Image.LANCZOS)
    real_cal = TableCalibration.load(fix / "calibration.json")
    results = {}
    timers = Timers()
    restore = [
        (calmod, "locate_table", timers.wrap(calmod, "locate_table", "table_detection")),
        (pnmod, "locate_table", timers.wrap(pnmod, "locate_table", "table_detection")),
        (pnmod, "crop", timers.wrap(pnmod, "crop", "region_extraction")),
        (textmod.TemplateOCR, "read_text",
         timers.wrap(textmod.TemplateOCR, "read_text", "ocr")),
        (cardmod.TemplateCardRecognizer, "recognize",
         timers.wrap(cardmod.TemplateCardRecognizer, "recognize", "card_recognition")),
        (cardmod.PokerNowCardRecognizer, "recognize",
         timers.wrap(cardmod.PokerNowCardRecognizer, "recognize", "card_recognition")),
        (StateTracker, "update", timers.wrap(StateTracker, "update", "state_tracker")),
    ]
    try:
        for name, cal_, imgs in (("synthetic_1280x800", cal, synth),
                                 ("pokernow_tuning_frame_2x (timing only)", real_cal, [real2x])):
            i = [0]

            class Scr:
                def capture(self):
                    t = time.perf_counter()
                    img = imgs[i[0] % len(imgs)].copy()      # what a capture returns
                    i[0] += 1
                    timers.cur["capture"] = timers.cur.get("capture", 0.0) + \
                        time.perf_counter() - t
                    return img
            s = LiveObserverSession(source_factory=lambda m, r: Scr())
            s.configure(CaptureSettings({"left": 0, "top": 0, "width": 0, "height": 0}, None),
                        cal_, 0.25, 0.5)
            stages = {}
            totals = []
            s.step()                                        # warm-up (templates, caches)
            timers.take()
            for _ in range(frames):
                t = time.perf_counter()
                s.step()
                step_t = time.perf_counter() - t
                c = timers.take()
                t = time.perf_counter()
                draw_overlay(s.last_frame, s.calibration, s.table_bbox(), s.last_observation)
                c["overlay_rendering"] = time.perf_counter() - t
                t = time.perf_counter()
                live_check(s, 0.5)
                c["decision_gate"] = time.perf_counter() - t
                totals.append(step_t + c["overlay_rendering"])
                for k, v in c.items():
                    stages.setdefault(k, []).append(v)
            res = {k: pct(v) for k, v in stages.items()}
            res["step_plus_overlay"] = pct(totals)
            results[name] = res
    finally:
        for owner, attr, orig in restore:
            setattr(owner, attr, orig)
    results["decision"] = _decision_timing()
    results["streamlit_rerun"] = _ui_timing()
    budget = {}
    for name in [k for k in results if k.startswith(("synthetic", "pokernow"))]:
        p99 = results[name]["step_plus_overlay"]["p99_ms"] / 1e3
        med = results[name]["step_plus_overlay"]["median_ms"] / 1e3
        ui = (results["streamlit_rerun"] or {}).get("median_ms", 0) / 1e3
        budget[name] = {f"{f:g}_fps": {"budget_ms": round(1e3 / f, 1),
                                        "median_share": round((med + ui) * f, 3),
                                        "p99_share": round((p99 + ui) * f, 3),
                                        "keeps_up_p99": (p99 + ui) * f < 1.0}
                        for f in FPS}
    out_d = {"what": "per-stage latency of one live observer step on a mocked screen",
             "machine_note": "container CPU; real mss capture time not measured (no display)",
             "frames": frames, "stages": results, "budget": budget,
             "decision_note": "decision runs only when the observed state changes "
                              "(cached per state), not every frame"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_d, indent=1))
    return out_d


def _decision_timing():
    from poker_alpha.decision import DecisionConfig, recommend_action

    cal = default_layout(2, 0)
    tr = StateTracker(cal, 0.5, 1.0)
    from poker_alpha.observer.pokernow import PokerNowStyleAdapter

    ad = PokerNowStyleAdapter(cal)
    img = render_table(SyntheticTable(seats=[SyntheticSeat("h", 99.0),
                                             SyntheticSeat("v", 96.0, bet=2.0)],
                                      dealer=1, hero_cards=("As", "Kd"),
                                      board=("Qs", "Jh", "4c"), pot=3.0, actor=0), cal)
    for _ in range(3):
        tr.update(ad.read_frame(img))
    st = tr.to_observed_state()
    out = {}
    for label, cfg in (("equity_2000_no_rollout", DecisionConfig(equity_simulations=2000)),
                       ("equity_2000_rollout_400", DecisionConfig(equity_simulations=2000,
                                                                  rollout_simulations=400))):
        ts = []
        for k in range(5):
            t = time.perf_counter()
            recommend_action(st, config=DecisionConfig(**{**cfg.__dict__, "seed": k}))
            ts.append(time.perf_counter() - t)
        out[label] = pct(ts)
    return out


def _ui_timing(reruns: int = 6):
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return None
    from observer_helpers import MON, Screen, frames_of, table

    at = AppTest.from_file(str(ROOT / "poker_alpha" / "ui" / "app.py"), default_timeout=300)
    scr = Screen(frames_of(table(), repeat=50))
    at.session_state["live_source_factory"] = lambda m, r: scr
    at.session_state["live_monitor_lister"] = lambda: [MON, MON]
    at.run()
    at.sidebar.radio[0].set_value("Live screen").run()
    at.checkbox(key="live_compute").set_value(False).run()
    [b for b in at.button if b.label == "Start Live Observer"][0].click().run()
    ts = []
    for _ in range(reruns):
        t = time.perf_counter()
        at.run()
        ts.append(time.perf_counter() - t)
    r = pct(ts)
    r["note"] = ("full script rerun in AppTest incl. one observer step, both images and the "
                 "tables; the live page reruns only the fragment, so this is an upper bound")
    return r


# -- soak ---------------------------------------------------------------------------

def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def soak(minutes: float, fps: float, out: Path, record: bool = True) -> dict:
    from poker_alpha.observer.session import TestSessionRecorder

    cal = default_layout(2, 0)
    rng = np.random.default_rng(0)
    cache = {}

    def frame(i):
        k = i // 3 % 64           # each table state is shown for 3 frames
        if k not in cache:
            cache[k] = render_table(_table(k), cal)
        r = rng.random()
        if r < 0.01:
            return None                                   # capture error
        if r < 0.02:
            return Image.new("RGB", (1280, 800), (0, 0, 0))   # blank / covered
        return cache[k]
    i = [0]

    class Scr:
        def capture(self):
            f = frame(i[0])
            i[0] += 1
            if f is None:
                raise OSError("simulated capture failure")
            return f
    s = LiveObserverSession(source_factory=lambda m, r: Scr())
    s.configure(CaptureSettings({"left": 0, "top": 0, "width": 1280, "height": 800}, None),
                cal, 0.5, 1.0)
    tmp = Path(tempfile.mkdtemp(prefix="pa_soak_"))
    rec = TestSessionRecorder(tmp) if record else None
    if rec:
        rec.start(cal, {"left": 0, "top": 0, "width": 1280, "height": 800}, None)
    t0 = time.time()
    samples = []
    next_sample = t0
    interval = 1.0 / fps
    late = 0
    steps = 0
    while time.time() - t0 < minutes * 60:
        t = time.time()
        res = s.step()
        if rec:
            rec.on_step(s, res)
        steps += 1
        if t >= next_sample:
            gc.collect()
            disk = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
            samples.append({"t_s": round(t - t0, 1), "steps": steps, "rss_mb": round(_rss_mb(), 1),
                            "gc_objects": len(gc.get_objects()),
                            "tracker_events": len(s.tracker.events),
                            "tracker_flags": len(s.tracker.flags),
                            "tracker_actions": len(s.tracker.actions),
                            "hand_number": s.tracker.hand_number,
                            "session_disk_mb": round(disk / 1e6, 2),
                            "retained": len(rec.samples) if rec else 0})
            next_sample += 30
        spent = time.time() - t
        if spent > interval:
            late += 1
        else:
            time.sleep(interval - spent)
    if rec:
        rec.finish()
    first, last = samples[1] if len(samples) > 1 else samples[0], samples[-1]
    hours = max(1e-9, (last["t_s"] - first["t_s"]) / 3600)
    result = {"minutes": minutes, "fps": fps, "steps": steps, "late_steps": late,
              "samples": samples,
              "growth_per_hour": {k: round((last[k] - first[k]) / hours, 2)
                                  for k in ("rss_mb", "gc_objects", "tracker_events",
                                            "tracker_flags", "session_disk_mb")},
              "session_dir": str(tmp)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("profile")
    a.add_argument("--frames", type=int, default=120)
    a.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "live_observer_perf.json")
    b = sub.add_parser("soak")
    b.add_argument("--minutes", type=float, default=30)
    b.add_argument("--fps", type=float, default=3.0)
    b.add_argument("--no-record", action="store_true")
    b.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "live_observer_soak.json")
    args = p.parse_args(argv)
    if args.cmd == "profile":
        r = profile(args.frames, args.out)
        for name, st in r["stages"].items():
            print(name)
            if not st:
                continue
            for k, v in st.items():
                if isinstance(v, dict) and "median_ms" in v:
                    print(f"  {k:<26} median {v['median_ms']:>8} ms  p95 {v['p95_ms']:>8}  "
                          f"p99 {v['p99_ms']:>8}")
        for name, b_ in r["budget"].items():
            print(name, {k: (v["p99_share"], v["keeps_up_p99"]) for k, v in b_.items()})
    else:
        r = soak(args.minutes, args.fps, args.out, not args.no_record)
        print(json.dumps(r["growth_per_hour"]), "steps", r["steps"], "late", r["late_steps"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
