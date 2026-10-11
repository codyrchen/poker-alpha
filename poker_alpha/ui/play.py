"""PokerAlpha Play Mode — the user-facing decision view.

Four inputs (Demo, Manual, Hand history, Live screen), one presentation:
current hand → recommendation (or a deliberate abstention) → confidence →
"Why this recommendation". Everything is computed by the existing pipeline
(`recommend_action`); this module only selects input and renders the view
models from :mod:`poker_alpha.ui.viewmodel`.

Diagnostic tooling (calibration editor, OCR tables, region debugger,
annotation) lives in Developer Mode, not here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import streamlit as st

from poker_alpha.decision import recommend_action
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.ui import components as C
from poker_alpha.ui.viewmodel import (hand_vm, live_status_vm,
                                      recommendation_vm, why_vm)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# demo spots — every one runs through the real pipeline; the labels state
# what actually happens (verified against the release strategy + gate).
# ---------------------------------------------------------------------------

def _hu(hero_cards, board, pot, actions, committed, stacks, bets=(0.0, 0.0)):
    return {
        "num_seats": 2, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": hero_cards, "board": board, "pot": pot, "actor": 0,
        "seats": [{"stack": stacks[0], "committed": committed[0], "bet": bets[0]},
                  {"stack": stacks[1], "committed": committed[1], "bet": bets[1]}],
        "actions": actions}


_PRE = [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
        {"street": 0, "seat": 1, "kind": "call", "amount": 1.5}]
_CHECKS_TO_RIVER = [{"street": 1, "seat": 1, "kind": "check"},
                    {"street": 1, "seat": 0, "kind": "check"},
                    {"street": 2, "seat": 1, "kind": "check"},
                    {"street": 2, "seat": 0, "kind": "check"}]

DEMO_SPOTS = {
    "River, checked to hero — solver accepted": _hu(
        "Ah Kd", "Qs 7d 2c Th 3h", 5.0,
        _PRE + _CHECKS_TO_RIVER + [{"street": 3, "seat": 1, "kind": "check"}],
        (2.5, 2.5), (97.5, 97.5)),
    "River, facing a bet — solver at low confidence (off-tree size)": _hu(
        "9h 9c", "Qs 7d 2c Th 3h", 7.5,
        _PRE + _CHECKS_TO_RIVER + [{"street": 3, "seat": 1, "kind": "bet",
                                    "amount": 2.5}],
        (2.5, 5.0), (97.5, 95.0), bets=(0.0, 2.5)),
    "Preflop open decision — gate rejects (seed disagreement)": _hu(
        "Ad Qs", "", 1.5, [], (0.5, 1.0), (99.5, 99.0), bets=(0.5, 1.0)),
    "Preflop vs 3-bet — rollout fallback": _hu(
        "Ts Td", "", 11.5,
        [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
         {"street": 0, "seat": 1, "kind": "raise", "amount": 9.0}],
        (2.5, 9.0), (97.5, 91.0), bets=(2.5, 9.0)),
    "6-max flop c-bet — multiway (rollout only)": {
        "num_seats": 6, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
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
    },
}


# ---------------------------------------------------------------------------
# the one Play renderer
# ---------------------------------------------------------------------------

def _compute(obs, cfg, conf):
    cfg = replace(cfg, observer_confidence=conf)
    key = json.dumps([obs.state_key() if hasattr(obs, "state_key") else repr(obs),
                      cfg.equity_simulations, cfg.rollout_simulations, cfg.seed],
                     default=str)
    cached = st.session_state.get("play_report")
    if cached and cached[0] == key:
        return cached[1]
    with st.spinner("Analysing…"):
        report = recommend_action(obs, config=cfg)
    st.session_state["play_report"] = (key, report)
    return report


def render_play_report(obs, cfg, conf=None, report=None, store=None) -> None:
    """Hand → recommendation/abstention → confidence → why."""
    if report is None:
        cfg2 = replace(cfg, observer_confidence=conf)
        with st.spinner("Analysing…"):
            report = recommend_action(obs, config=cfg2)
        if store is not None:
            store(report)
    C.render_hand(hand_vm(obs))
    vm = recommendation_vm(report)
    if vm.refusal is not None:
        C.render_refusal(vm.refusal)
        return
    if vm.abstention is not None:
        C.render_abstention(vm.abstention)
        if vm.abstention.withheld and vm.recommended is None:
            return
    C.render_actions(vm)
    C.render_status(vm)
    C.render_why(why_vm(report), report)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _demo(cfg) -> None:
    name = st.selectbox("Demo hand", list(DEMO_SPOTS), key="play_demo_spot")
    st.caption("Every demo runs through the real pipeline: the 2M release "
               "strategy, the confidence gate, rollouts and heuristics. "
               "Nothing is pre-recorded.")
    obs = ManualStateAdapter.from_dict(DEMO_SPOTS[name])
    render_play_report(obs, cfg)


def _manual(cfg) -> None:
    """Simple heads-up hand entry; the full JSON editor stays available."""
    use_json = st.toggle("Advanced JSON state", value=False, key="play_use_json")
    if use_json:
        default = json.dumps(
            DEMO_SPOTS["River, checked to hero — solver accepted"], indent=1)
        text = st.text_area("Observed state (pokeralpha.observed/v1 JSON)",
                            default, height=260, key="play_manual")
        try:
            obs = ManualStateAdapter.from_dict(json.loads(text))
        except (ValueError, KeyError, TypeError) as exc:
            C.render_waiting("The entered state could not be parsed")
            st.caption(f"Details: {exc}")
            return
        render_play_report(obs, cfg)
        return
    st.caption("Heads-up spot, blinds 0.5 / 1 — amounts in big blinds. "
               "Use Advanced JSON state for anything else.")
    c = st.columns(4)
    hero_cards = c[0].text_input("Hero cards", "As Ks", key="pm_hero")
    board = c[1].text_input("Board (0/3/4/5 cards)", "Qs Js 4h", key="pm_board")
    position = c[2].selectbox("Hero position", ["BTN", "BB"], key="pm_pos")
    pot = c[3].number_input("Pot (BB)", min_value=0.0, value=5.0, step=0.5,
                            key="pm_pot")
    c = st.columns(4)
    to_call = c[0].number_input("To call (BB)", min_value=0.0, value=0.0,
                                step=0.5, key="pm_tocall")
    hero_stack = c[1].number_input("Hero stack (BB)", min_value=0.0,
                                   value=97.5, step=0.5, key="pm_hstack")
    opp_stack = c[2].number_input("Opponent stack (BB)", min_value=0.0,
                                  value=97.5, step=0.5, key="pm_ostack")
    hero_committed = c[3].number_input(
        "Hero already in pot (BB)", min_value=0.0, value=2.5, step=0.5,
        key="pm_committed",
        help="Chips the hero has put in this hand — lets the solver map the "
             "spot onto its trained 100 BB tree.")
    dealer = 0 if position == "BTN" else 1
    opp_committed = max(0.0, pot - hero_committed)
    state = {
        "num_seats": 2, "hero_seat": 0, "dealer": dealer,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": hero_cards, "board": board, "pot": pot, "actor": 0,
        "seats": [{"stack": hero_stack, "committed": hero_committed},
                  {"stack": opp_stack, "committed": opp_committed,
                   "bet": to_call}],
    }
    try:
        obs = ManualStateAdapter.from_dict(state)
    except (ValueError, KeyError, TypeError) as exc:
        C.render_waiting("The entered hand could not be built")
        st.caption(f"Details: {exc}")
        return
    render_play_report(obs, cfg)


def _replay(cfg) -> None:
    from poker_alpha.history import load_hands, parse_hands, replay_hand

    up = st.file_uploader("pokeralpha.hand/v1 JSON", type=["json"],
                          key="play_hands")
    if up is not None:
        hands = parse_hands(json.loads(up.read()))
    else:
        st.caption("Using the bundled sample hand history.")
        hands = load_hands(FIXTURES / "hands" / "sample.json")
    c1, c2 = st.columns(2)
    h = c1.selectbox("Hand", range(len(hands)),
                     format_func=lambda i: f"hand {i}", key="play_hand_i")
    res = replay_hand(hands[h], hand_index=h)
    if not res.decisions:
        C.render_waiting("No hero decisions in this hand")
        return
    d = c2.selectbox("Hero decision", range(len(res.decisions)),
                     format_func=lambda i: f"{res.decisions[i].state.street.name}: "
                                           f"hero {res.decisions[i].action}",
                     key="play_dec_i")
    dec = res.decisions[d]
    st.caption(f"Actual action in the history: {dec.action}"
               + ("" if dec.amount is None else f" to {dec.amount:g}"))
    render_play_report(dec.state, cfg)


def _live(cfg) -> None:
    """Slim live view: capture status, recognized hand, recommendation.

    Reuses the observer session machinery; the full calibration editor,
    overlays, OCR tables and the region debugger stay in Developer Mode.
    """
    from poker_alpha.observer.errors import ObserverDependencyError
    from poker_alpha.ui import live_panel as LP
    from poker_alpha.observer.live import live_check
    from poker_alpha.observer.pokernow import pokernow_hu_layout

    st.caption("Reads pixels only — never clicks, types or acts. Use only in "
               "private, play-money or test games where real-time assistance "
               "is permitted.")
    from poker_alpha.observer.live import CaptureSettings, list_monitors

    try:
        session = LP._session()
    except ObserverDependencyError as exc:
        C.render_waiting("Screen capture is not available")
        st.caption(str(exc) + " — install the [vision] extra.")
        return
    with st.expander("Capture & table setup", expanded=not session.running):
        lister = st.session_state.get("live_monitor_lister", list_monitors)
        try:
            monitors = [m for m in lister() if m["index"] >= 1] or lister()
        except Exception as exc:  # noqa: BLE001
            C.render_waiting("Screen capture is not available")
            st.caption(f"{type(exc).__name__}: {exc}")
            return
        labels = [f"monitor {m['index']}: {m['width']}x{m['height']}"
                  for m in monitors]
        c = st.columns(4)
        i = c[0].selectbox("Monitor", range(len(monitors)),
                           format_func=lambda k: labels[k], key="play_monitor")
        side = c[1].radio("Hero plate", ["right", "left"], horizontal=True,
                          key="play_hero_side")
        sb = c[2].number_input("Small blind", value=0.5, key="play_sb")
        bb = c[3].number_input("Big blind", value=1.0, key="play_bb")
        mon = monitors[i]
        with st.expander("Advanced capture settings"):
            st.caption("Capture rectangle relative to the monitor, in screen "
                       "points (width or height 0 = whole monitor). Crop to "
                       "the PokerNow table.")
            cc = st.columns(4)
            left = cc[0].number_input("left", 0, max(0, mon["width"] - 1), 0,
                                      key="play_cap_left")
            top = cc[1].number_input("top", 0, max(0, mon["height"] - 1), 0,
                                     key="play_cap_top")
            width = cc[2].number_input("width", 0, mon["width"], 0,
                                       key="play_cap_width")
            height = cc[3].number_input("height", 0, mon["height"], 0,
                                        key="play_cap_height")
        rect = (int(left), int(top), int(width), int(height)) \
            if width and height else None
        capture = CaptureSettings(mon, rect)
        st.caption("Uses the PokerNow heads-up layout preset. For custom "
                   "calibration, use Developer Mode → Live screen.")
    cal = pokernow_hu_layout(hero_side=side)
    session.configure(capture, cal, sb, bb)
    b = st.columns(3)
    if b[0].button("Start", disabled=session.running, key="play_start"):
        session.start()
        st.rerun()
    if b[1].button("Stop", disabled=not session.running, key="play_stop"):
        session.stop()
        st.rerun()
    if b[2].button("Read one frame", key="play_once"):
        session.step()

    def view():
        if session.running:
            session.step()
        tracker = session.tracker
        check = live_check(session, 0.5)
        tracked = tracker.tracked()
        status = live_status_vm(check.ok, check.critical_confidence,
                                check.problems, paused=tracked.paused)
        solver_trust = None
        cached = st.session_state.get("play_live_report")
        if not status.ok:
            C.render_live_status(status, None)
            C.render_waiting(status.waiting_for or "Waiting for a reliable table read")
            with st.expander("Technical details"):
                for p in check.problems:
                    st.caption(p)
            return
        obs = tracker.to_observed_state()
        report = _compute(obs, cfg, check.critical_confidence)
        vm = recommendation_vm(report)
        C.render_live_status(status, vm.confidence)
        render_play_report(obs, cfg, check.critical_confidence, report=report)

    fragment = getattr(st, "fragment", None)
    if fragment is None or not session.running:
        view()
    else:
        fragment(run_every=1.0)(view)()


def play_mode(cfg) -> None:
    C.inject_styles()
    sub = st.radio("Input", ["Demo", "Manual", "Hand history", "Live screen"],
                   horizontal=True, key="play_input", label_visibility="collapsed")
    if sub == "Demo":
        _demo(cfg)
    elif sub == "Manual":
        _manual(cfg)
    elif sub == "Hand history":
        _replay(cfg)
    else:
        _live(cfg)
