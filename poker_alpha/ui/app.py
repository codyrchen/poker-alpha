"""PokerAlpha local decision-support UI.

    streamlit run poker_alpha/ui/app.py

Display only: it shows analysis and never clicks, bets or controls a
browser. Real-time use is only for private / play-money / test games where
assistance is permitted; otherwise use the hand-history and screenshot
modes after the session.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.holdem import ManualStateAdapter, validate
from poker_alpha.ui.view import (candidate_rows, headline, range_rows,
                                 seat_rows, solver_signal_rows, solver_status,
                                 source_rows, state_rows, uncertainty_rows)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STRATEGY = ROOT / "results" / "strategy" / "holdem_v2_seed0.npz"

EXAMPLE_STATE = {
    "format": "pokeralpha.observed/v1", "num_seats": 6, "hero_seat": 0,
    "dealer": 0, "small_blind": 0.5, "big_blind": 1.0,
    "hero_cards": "As Ks", "board": "Qs Js 4h", "pot": 13.5, "actor": 0,
    "seats": [{"stack": 96.5}, {"stack": 99.5, "folded": True},
              {"stack": 90.0}, {"stack": 100, "folded": True},
              {"stack": 100, "folded": True}, {"stack": 100, "folded": True}],
    "actions": [
        {"street": 0, "seat": 3, "kind": "fold"},
        {"street": 0, "seat": 4, "kind": "fold"},
        {"street": 0, "seat": 5, "kind": "fold"},
        {"street": 0, "seat": 0, "kind": "raise", "amount": 3.5},
        {"street": 0, "seat": 1, "kind": "fold"},
        {"street": 0, "seat": 2, "kind": "call", "amount": 2.5},
        {"street": 1, "seat": 2, "kind": "check"}],
}


def sidebar_config() -> DecisionConfig:
    st.sidebar.header("Analysis settings")
    eq = st.sidebar.slider("Equity simulations", 200, 10000, 2000, step=200)
    ro = st.sidebar.slider("Rollout simulations (0 = heuristic only)", 0, 5000,
                           1000, step=100)
    seed = st.sidebar.number_input("Seed", value=0, step=1)
    use_solver = st.sidebar.checkbox("Use HU solver strategy (gated)", value=True)
    solver, unavailable = None, None
    if use_solver:
        prov = _load_solver(str(STRATEGY))
        if hasattr(prov, "code"):
            unavailable = prov
        else:
            solver = prov
    return DecisionConfig(equity_simulations=eq, rollout_simulations=ro,
                          seed=int(seed), solver=solver, solver_unavailable=unavailable)


@st.cache_resource
def _load_solver(path: str):
    from poker_alpha.pipeline import load_solver
    return load_solver(path)


def manual_state():
    st.subheader("Manual state entry")
    text = st.text_area("Observed state (pokeralpha.observed/v1 JSON)",
                        json.dumps(EXAMPLE_STATE, indent=1), height=260)
    try:
        return ManualStateAdapter.from_dict(json.loads(text)), None
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Cannot parse state: {exc}")
        return None, None


def replay_state():
    from poker_alpha.history import load_hands, parse_hands, replay_hand

    st.subheader("Hand-history replay (post-hand analysis)")
    up = st.file_uploader("pokeralpha.hand/v1 JSON", type=["json"])
    if up is not None:
        hands = parse_hands(json.loads(up.read()))
    else:
        st.caption("Using bundled sample: tests/fixtures/hands/sample.json")
        hands = load_hands(FIXTURES / "hands" / "sample.json")
    h = st.selectbox("Hand", range(len(hands)), format_func=lambda i: f"hand {i}")
    res = replay_hand(hands[h], hand_index=h)
    if not res.decisions:
        st.info("No hero decisions in this hand.")
        return None, None
    d = st.selectbox("Hero decision", range(len(res.decisions)),
                     format_func=lambda i: f"{res.decisions[i].state.street.name}: "
                                           f"hero {res.decisions[i].action}")
    dec = res.decisions[d]
    st.write(f"Actual hero action: **{dec.action}**"
             + ("" if dec.amount is None else f" to {dec.amount:g}"))
    return dec.state, None


def observer_state():
    from PIL import Image

    from poker_alpha.observer.calibration import TableCalibration
    from poker_alpha.observer.fusion import StateTracker
    from poker_alpha.observer.pokernow import PokerNowStyleAdapter

    st.subheader("Screen observer (screenshots)")
    st.caption("Exact PokerNow visual accuracy is not validated without "
               "representative screenshots. Use live capture only where "
               "real-time assistance is permitted.")
    cal_path = st.text_input("Calibration JSON",
                             str(FIXTURES / "table_calibration.json"))
    c1, c2 = st.columns(2)
    sb = c1.number_input("Small blind", value=0.5)
    bb = c2.number_input("Big blind", value=1.0)
    cal = TableCalibration.load(cal_path)
    if "tracker" not in st.session_state or st.session_state.get("tracker_cal") != cal_path:
        st.session_state.tracker = StateTracker(cal, sb, bb)
        st.session_state.tracker_cal = cal_path
    tracker: StateTracker = st.session_state.tracker
    adapter = PokerNowStyleAdapter(cal)

    up = st.file_uploader("Screenshot (PNG)", type=["png", "jpg"])
    b1, b2, b3 = st.columns(3)
    if b1.button("Pause observer"):
        tracker.pause()
    if b2.button("Resume"):
        tracker.resume()
    frames = b3.number_input("Frames to ingest", 1, 10, 3)
    if st.button("Process screenshot"):
        img = Image.open(up if up is not None else FIXTURES / "table.png").convert("RGB")
        for _ in range(int(frames)):
            tracker.update(adapter.read_frame(img))
    with st.expander("Correct a recognized value"):
        name = st.selectbox("Field", sorted(tracker.fields))
        value = st.text_input("Value (JSON, e.g. 13.5, \"As\", true, null)")
        if st.button("Apply correction") and value:
            tracker.correct(name, json.loads(value))
        if st.button("Release correction"):
            tracker.release(name)
    tracked = tracker.tracked()
    st.write(f"Observer {'PAUSED' if tracked.paused else 'running'} · hand "
             f"{tracked.hand_number} · critical confidence "
             f"{tracker.critical_confidence():.0%}")
    for w in tracked.warnings[-5:]:
        st.warning(w)
    if tracked.snapshot.hero_cards[0] is None:
        st.info("No confirmed hero cards yet: process the screenshot (≥3 frames).")
        return None, None
    return tracker.to_observed_state(), tracker.critical_confidence()


def render_report(obs, cfg, conf, report=None, store=None) -> None:
    """Spot, validation and DecisionReport for an observed state."""
    from dataclasses import replace

    cfg = replace(cfg, observer_confidence=conf)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Spot")
        st.table(state_rows(obs))
        st.dataframe(seat_rows(obs), hide_index=True)
        for issue in validate(obs):
            (st.error if issue.severity == "error" else st.warning)(issue.message)
    with right:
        if report is None:
            with st.spinner("Analysing..."):
                report = recommend_action(obs, config=cfg)
            if store is not None:
                store(report)
        st.subheader(headline(report))
        m1, m2, m3 = st.columns(3)
        m1.metric("Hero equity", "?" if report.hero_equity is None else
                  f"{report.hero_equity:.1%}",
                  None if not report.hero_equity_se else f"±{report.hero_equity_se:.1%}")
        m2.metric("Pot odds", "-" if report.pot_odds is None else f"{report.pot_odds:.1%}")
        m3.metric("SPR", "-" if report.spr is None else f"{report.spr:.2f}")
        st.dataframe(candidate_rows(report), hide_index=True)
        st.caption(f"Frequencies: {report.mix_meaning}")
        st.markdown(f"**{solver_status(report)}**")
        if solver_signal_rows(report):
            st.dataframe(solver_signal_rows(report), hide_index=True)
        st.markdown("**Decision sources (solver -> rollout -> heuristic)**")
        st.dataframe(source_rows(report), hide_index=True)
        st.markdown("**Uncertainty by source**")
        st.dataframe(uncertainty_rows(report), hide_index=True)
        st.markdown("**Opponent ranges (beliefs, not known hands)**")
        st.dataframe(range_rows(report), hide_index=True)
        for w in report.warnings:
            st.warning(w)


def main() -> None:
    st.set_page_config(page_title="PokerAlpha", layout="wide")
    st.title("PokerAlpha decision support")
    st.caption("Analysis only — PokerAlpha never clicks, bets or acts for you.")
    cfg = sidebar_config()
    mode = st.sidebar.radio("Input", ["Manual entry", "Hand-history replay",
                                      "Screen observer", "Live screen"])
    if mode == "Live screen":
        from poker_alpha.ui.live_panel import live_screen_mode
        live_screen_mode(cfg, render_report)
        return
    if mode == "Manual entry":
        obs, conf = manual_state()
    elif mode == "Hand-history replay":
        obs, conf = replay_state()
    else:
        obs, conf = observer_state()
    if obs is None:
        return
    render_report(obs, cfg, conf)


main()
