"""Streamlit "Live screen" mode: live capture, calibration overlay, fused
state, gated decision report.

Read-only: frames are captured and analysed locally; nothing is clicked,
typed, controlled or submitted, and nothing is uploaded. A frame is written
to disk only when "Save current frame" is pressed. Use live mode only in
private, play-money or test games where real-time assistance is permitted;
otherwise use the screenshot or hand-history modes after the session.

The capture loop is a Streamlit fragment with ``run_every`` (one frame per
rerun of the fragment), not a blocking loop; the same ``LiveObserverSession``
(and its ``StateTracker``) lives in ``st.session_state`` across frames.
Tests inject a fake screen via ``st.session_state["live_source_factory"]``
and ``st.session_state["live_monitor_lister"]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ..observer.calibration import TableCalibration
from ..observer.errors import ObserverDependencyError
from ..observer.live import (CaptureSettings, LiveObserverSession, critical_check,
                             draw_overlay, fused_rows, list_monitors, mss_source_factory,
                             raw_rows, save_frame, seat_state_rows, transform_regions,
                             with_region)
from ..observer.pokernow import LAYOUT_PRESETS, default_layout, pokernow_hu_layout
from ..observer.regions import Region

DEFAULT_SAVE_DIR = "~/pokeralpha_captures"


def _image(img, caption):
    try:
        st.image(img, caption=caption, width="stretch")
    except (TypeError, ValueError):            # older Streamlit
        st.image(img, caption=caption, use_container_width=True)


def _session() -> LiveObserverSession:
    if "live_session" not in st.session_state:
        factory = st.session_state.get("live_source_factory", mss_source_factory)
        st.session_state.live_session = LiveObserverSession(source_factory=factory)
    return st.session_state.live_session


def _calibration() -> TableCalibration:
    if "live_cal" not in st.session_state:
        st.session_state.live_cal = default_layout(6, 0)
    return st.session_state.live_cal


def _set_cal(cal: TableCalibration) -> None:
    """Replace the calibration and drop editor widgets that hold old values."""
    st.session_state.live_cal = cal
    for k in list(st.session_state.keys()):
        if str(k).startswith(("live_r", "live_dx", "live_dy", "live_sx", "live_sy", "live_bx", "live_by",
                                "live_bbox_mode", "live_felt")) \
                and k != "live_report":
            del st.session_state[k]


# -- capture controls -----------------------------------------------------------

def _capture_controls(session: LiveObserverSession):
    st.markdown("#### 1. Capture")
    lister = st.session_state.get("live_monitor_lister", list_monitors)
    try:
        monitors = [m for m in lister() if m["index"] >= 1] or lister()
    except ObserverDependencyError as exc:
        st.error(f"{exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot list monitors: {type(exc).__name__}: {exc}")
        return None
    labels = [f"monitor {m['index']}: {m['width']}x{m['height']} at ({m['left']},{m['top']})"
              for m in monitors]
    i = st.selectbox("Monitor", range(len(monitors)), format_func=lambda k: labels[k],
                     key="live_monitor")
    mon = monitors[i]
    st.caption("Capture rectangle relative to the monitor, in screen points "
               "(width or height 0 = whole monitor). Crop to the PokerNow table.")
    c = st.columns(4)
    left = c[0].number_input("left", 0, max(0, mon["width"] - 1), 0, key="live_left")
    top = c[1].number_input("top", 0, max(0, mon["height"] - 1), 0, key="live_top")
    width = c[2].number_input("width", 0, mon["width"], 0, key="live_width")
    height = c[3].number_input("height", 0, mon["height"], 0, key="live_height")
    rect = (int(left), int(top), int(width), int(height)) if width and height else None
    return CaptureSettings(mon, rect)


# -- calibration editor -------------------------------------------------------------

def _calibration_editor(session: LiveObserverSession):
    st.markdown("#### 2. Calibration")
    cal = _calibration()
    with st.expander("Calibration / debug workflow", expanded=False):
        st.markdown(
            "1. Crop the capture to the table and press **Capture one frame**.\n"
            "2. Set the table bounds (fixed box, or felt colour detection).\n"
            "3. Shift / scale all regions until the coloured boxes sit on the table.\n"
            "4. Fine-tune single regions, check the raw readings, save the JSON.\n"
            "Disable decisions while calibrating.")
        preset = st.radio("Layout", LAYOUT_PRESETS, horizontal=True, key="live_preset",
                          index=1 if cal.client == "pokernow" else 0)
        if preset == "PokerNow Heads-Up":
            st.caption("Measured on a real PokerNow heads-up table: both players along the "
                       "bottom edge, seat 0 = hero, seat 1 = opponent. Table bounds = the green "
                       "felt, found every frame. Board slots are not yet verified on a real board.")
            c1, c2 = st.columns(2)
            side = c1.radio("Hero plate on screen", ["right", "left"], horizontal=True,
                            key="live_hu_side")
            if c2.button("Use PokerNow Heads-Up layout"):
                _set_cal(pokernow_hu_layout(side))
                st.rerun()
        else:
            c1, c2, c3 = st.columns(3)
            seats = c1.number_input("Seats", 2, 9, cal.num_seats, key="live_seats")
            hero = c2.number_input("Hero seat", 0, int(seats) - 1,
                                   min(cal.hero_seat, int(seats) - 1), key="live_hero")
            if c3.button("Use default layout"):
                _set_cal(default_layout(int(seats), int(hero)))
                st.rerun()
        st.caption(f"Active calibration: **{cal.name}** ({cal.num_seats} seats, hero seat "
                   f"{cal.hero_seat}, recognizers: {cal.client})")
        path = st.text_input("Calibration JSON path", "pokernow_calibration.json", key="live_cal_path")
        b1, b2 = st.columns(2)
        if b1.button("Load calibration"):
            try:
                _set_cal(TableCalibration.load(Path(path).expanduser()))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Cannot load {path}: {exc}")
        if b2.button("Save calibration"):
            try:
                cal.save(Path(path).expanduser())
                st.success(f"Saved {Path(path).expanduser()}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Cannot save: {exc}")

        st.markdown("**Table bounds** (pixels of the captured frame)")
        modes = ["Detect felt colour", "PokerNow felt (hue)", "Fixed box"]
        cur_mode = 2 if cal.table_bbox is not None else (1 if cal.table_detector == "green_oval" else 0)
        mode = st.radio("Bounds", modes, index=cur_mode, horizontal=True, key="live_bbox_mode")
        if mode == "PokerNow felt (hue)":
            cal.table_bbox = None
            cal.table_detector = "green_oval"
            st.caption("Ellipse fitted to the largest felt-hue blob: follows window size and "
                       "browser zoom; ignores the felt's shading, logo and pot pill.")
        elif mode == "Fixed box":
            cur = cal.table_bbox or session.table_bbox() or (0, 0, 800, 500)
            c = st.columns(4)
            l_ = c[0].number_input("x0", 0, 20000, int(cur[0]), key="live_bx0")
            t_ = c[1].number_input("y0", 0, 20000, int(cur[1]), key="live_by0")
            r_ = c[2].number_input("x1", 1, 20000, int(cur[2]), key="live_bx1")
            b_ = c[3].number_input("y1", 1, 20000, int(cur[3]), key="live_by1")
            if (l_, t_, r_, b_) != tuple(cal.table_bbox or ()) and r_ > l_ and b_ > t_:
                cal.table_bbox = (int(l_), int(t_), int(r_), int(b_))
        else:
            cal.table_bbox = None
            cal.table_detector = "felt_color"
            hexcol = "#%02x%02x%02x" % tuple(cal.felt_color)
            col = st.color_picker("Felt colour", hexcol, key="live_felt")
            cal.felt_color = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
            cal.felt_tolerance = st.slider("Felt tolerance", 5, 120, int(cal.felt_tolerance),
                                           key="live_felt_tol")
            if session.last_frame is not None and st.button("Sample felt colour at table centre"):
                W, H = session.last_frame.size
                cal.felt_color = tuple(session.last_frame.getpixel((W // 2, int(H * 0.3))))[:3]
                st.rerun()

        st.markdown("**Coarse alignment** (move/scale every region; table-normalized units)")
        c = st.columns(5)
        dx = c[0].number_input("dx", -0.5, 0.5, 0.0, 0.005, format="%.3f", key="live_dx")
        dy = c[1].number_input("dy", -0.5, 0.5, 0.0, 0.005, format="%.3f", key="live_dy")
        sx = c[2].number_input("scale x", 0.5, 1.5, 1.0, 0.01, key="live_sx")
        sy = c[3].number_input("scale y", 0.5, 1.5, 1.0, 0.01, key="live_sy")
        if c[4].button("Apply"):
            try:
                _set_cal(transform_regions(cal, dx, dy, sx, sy))
                st.rerun()
            except ValueError as exc:
                st.error(f"Transform moves a region outside the table: {exc}")

        st.markdown("**Fine-tune one region**")
        name = st.selectbox("Region", sorted(cal.regions), key="live_region")
        r = cal.regions[name]
        c = st.columns(5)
        x = c[0].number_input("x", -0.5, 1.5, float(r.x), 0.002, format="%.3f", key=f"live_rx_{name}")
        y = c[1].number_input("y", -0.5, 1.5, float(r.y), 0.002, format="%.3f", key=f"live_ry_{name}")
        w = c[2].number_input("w", 0.001, 1.5, float(r.w), 0.002, format="%.3f", key=f"live_rw_{name}")
        h = c[3].number_input("h", 0.001, 1.5, float(r.h), 0.002, format="%.3f", key=f"live_rh_{name}")
        if c[4].button("Update"):
            try:
                _set_cal(with_region(cal, name, Region(x, y, w, h)))
                st.rerun()
            except ValueError as exc:
                st.error(f"Invalid region: {exc}")
    return _calibration()


# -- live view (fragment) -------------------------------------------------------------

def _decision(session, cfg, compute: bool, min_conf: float, render_report):
    tracker = session.tracker
    check = critical_check(tracker, min_conf)
    st.metric("Critical confidence", f"{check.critical_confidence:.0%}")
    if not check.ok:
        st.error("No decision: critical state validation failed — " + "; ".join(check.problems))
        return
    if not compute:
        st.info("Critical state valid. Decision computation is disabled (calibration mode).")
        return
    obs = tracker.to_observed_state()
    key = json.dumps([repr(obs.seats), obs.board, obs.hero_cards, obs.pot_total, obs.dealer,
                      obs.actor, repr(obs.action_history), cfg.equity_simulations,
                      cfg.rollout_simulations, cfg.seed], default=str)
    cached = st.session_state.get("live_report")
    report = cached[1] if cached and cached[0] == key else None
    render_report(obs, cfg, check.critical_confidence, report=report,
                  store=lambda rep: st.session_state.__setitem__("live_report", (key, rep)))


def _live_view(cfg, compute: bool, min_conf: float, save_dir: str, render_report):
    session = _session()
    if session.running:
        session.step()
    status = "RUNNING" if session.running else "stopped"
    st.caption(f"Observer {status} · frames {session.frames} · capture errors {session.errors} · "
               f"hand {session.tracker.hand_number}")
    if session.last_error:
        st.error(session.last_error)
    if session.last_warning:
        st.warning(session.last_warning)
    if session.last_frame is None:
        st.info("No frame yet: press Start Live Observer or Capture one frame.")
        return
    frame = session.last_frame
    c1, c2 = st.columns(2)
    with c1:
        _image(frame, f"raw captured frame {frame.size[0]}x{frame.size[1]} px")
    with c2:
        _image(draw_overlay(frame, session.calibration, session.table_bbox(),
                            session.last_observation),
               "calibration overlay: yellow table, magenta hero cards, cyan board, orange pot, "
               "green stacks, red bets, white dealer, blue seat")
    if st.button("Save current frame", key="live_save"):
        try:
            p = save_frame(frame, save_dir, {"capture": session.capture.__dict__ if session.capture else None,
                                             "calibration": session.calibration.name,
                                             "note": "local frame for fixture annotation"})
            st.success(f"Saved {p} (local only)")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cannot save frame: {exc}")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Raw recognition (last frame, before fusion)**")
        st.dataframe(raw_rows(session.last_observation), hide_index=True, height=320)
    with r2:
        st.markdown("**Fused StateTracker state**")
        st.dataframe(fused_rows(session.tracker), hide_index=True)
        st.dataframe(seat_state_rows(session.tracker), hide_index=True)
        for w in session.tracker.tracked().warnings[-5:]:
            st.warning(w)
    _decision(session, cfg, compute, min_conf, render_report)


def live_screen_mode(cfg, render_report) -> None:
    st.subheader("Live screen observer (read-only)")
    st.caption("Reads pixels only — never clicks, types or acts. Frames stay on this "
               "machine. Use only where real-time assistance is permitted.")
    session = _session()
    capture = _capture_controls(session)
    if capture is None:
        return
    cal = _calibration_editor(session)
    st.markdown("#### 3. Run")
    c = st.columns(3)
    sb = c[0].number_input("Small blind", value=0.5, key="live_sb")
    bb = c[1].number_input("Big blind", value=1.0, key="live_bb")
    fps = c[2].slider("Frames per second", 0.5, 3.0, 1.0, 0.5, key="live_fps")
    c = st.columns(2)
    compute = c[0].checkbox("Compute decisions", value=True, key="live_compute",
                            help="Turn off while calibrating the observer.")
    min_conf = c[1].slider("Minimum critical confidence for a decision", 0.0, 1.0, 0.5, 0.05,
                           key="live_min_conf")
    save_dir = st.text_input("Save frames to (local folder)", DEFAULT_SAVE_DIR, key="live_save_dir")
    session.configure(capture, cal, sb, bb)
    b = st.columns(4)
    if b[0].button("Start Live Observer", disabled=session.running):
        session.start()
        st.rerun()
    if b[1].button("Stop Live Observer", disabled=not session.running):
        session.stop()
        st.rerun()
    if b[2].button("Capture one frame"):
        session.step()
    if b[3].button("Reset tracker"):
        session.reset_tracker()
        st.session_state.pop("live_report", None)

    fragment = getattr(st, "fragment", None)
    run_every = (1.0 / float(fps)) if session.running else None
    if fragment is None:                         # Streamlit < 1.37: manual refresh only
        st.warning("This Streamlit version has no fragments: use Capture one frame.")
        _live_view(cfg, compute, min_conf, save_dir, render_report)
    else:
        fragment(run_every=run_every)(_live_view)(cfg, compute, min_conf, save_dir, render_report)
