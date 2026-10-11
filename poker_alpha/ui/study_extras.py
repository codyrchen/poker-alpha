"""Study Mode extras: Hand Replay (with modelled-range and decision-path
views) and Model Sensitivity. Pure orchestration of poker_alpha.analysis;
rendering reuses the Play components."""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from poker_alpha.analysis.decision_path import decision_path
from poker_alpha.analysis.ranges_view import range_stages
from poker_alpha.analysis.sensitivity import analyze_under_models
from poker_alpha.history import parse_hands
from poker_alpha.history.importers import (ImportFormatError,
                                           parse_external_history)

from .play import render_play_report

ROOT = Path(__file__).resolve().parents[2]
DEMO_SESSION = ROOT / "poker_alpha" / "data" / "demo_session.json"


# ---------------------------------------------------------------------------
# Hand Replay
# ---------------------------------------------------------------------------

def _load_hands_any(raw: bytes):
    """Canonical JSON or any registered external importer."""
    text = raw.decode("utf-8", errors="replace")
    try:
        return parse_hands(json.loads(text))
    except (json.JSONDecodeError, ValueError, KeyError):
        return parse_external_history(text)


def _range_matrix_html(stage) -> str:
    rows = []
    peak = max((max(r) for r in stage.matrix), default=0.0) or 1.0
    from poker_alpha.analysis.explorer import hand_classes

    grid = hand_classes()
    for i in range(13):
        tds = []
        for j in range(13):
            w = stage.matrix[i][j]
            a = min(w / peak, 1.0)
            tds.append(
                f'<td title="{grid[i][j]}: {w:.2%}" style="background:'
                f'rgba(42,120,214,{0.08 + 0.72 * a:.2f})">{grid[i][j]}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return ('<div style="overflow-x:auto"><table style="border-collapse:'
            'separate;border-spacing:1px;font-size:0.6rem;line-height:1.5;'
            'text-align:center">' + "".join(rows) + "</table></div>")


def replay_tab(cfg) -> None:
    up = st.file_uploader(
        "Session (pokeralpha.hand/v1 JSON or pokeralpha-lines/v1 text)",
        type=["json", "txt"], key="hr_upload")
    if up is not None:
        raw = up.read()
    else:
        st.caption("Using the bundled demo session. Imports: canonical JSON "
                   "or the pokeralpha-lines/v1 text format; other site "
                   "exports are future work pending real sample files.")
        raw = DEMO_SESSION.read_bytes()
    try:
        hands = _load_hands_any(raw)
    except (ImportFormatError, ValueError) as exc:
        st.error(str(exc))
        return
    # cache the analysed decisions per upload content
    key = ("hr", hash(raw), cfg.equity_simulations, cfg.rollout_simulations,
           cfg.seed)
    cached = st.session_state.get("hr_review")
    if cached and cached[0] == key:
        rev = cached[1]
    else:
        from poker_alpha.analysis import review_hands

        with st.spinner("Analysing every hero decision…"):
            rev = review_hands(hands, config=cfg)
        st.session_state["hr_review"] = (key, rev)

    if not rev.records:
        st.info("No hero decisions in this session.")
        return
    ss = st.session_state
    ss.setdefault("hr_i", 0)
    ss["hr_i"] = max(0, min(ss["hr_i"], len(rev.records) - 1))
    c1, c2, c3 = st.columns([1, 3, 1])
    if c1.button("← Previous decision", key="hr_prev",
                 disabled=ss["hr_i"] == 0):
        ss["hr_i"] -= 1
        st.rerun()
    if c3.button("Next decision →", key="hr_next",
                 disabled=ss["hr_i"] >= len(rev.records) - 1):
        ss["hr_i"] += 1
        st.rerun()
    r = rev.records[ss["hr_i"]]
    c2.markdown(f"<div style='text-align:center'>decision "
                f"<b>{ss['hr_i'] + 1}</b> of {len(rev.records)} — hand "
                f"{r.hand_index}, {r.street}</div>", unsafe_allow_html=True)

    st.markdown(f"**You chose `{r.actual_label or r.actual_kind}`**"
                + (f" (to {r.actual_amount:g})" if r.actual_amount else "")
                + f" — {r.regret.display()} · {r.alignment.display()}")
    if r.mapping_quality != "EXACT":
        st.caption(f"action mapping: {r.mapping_quality.lower()}")
    render_play_report(r.state, cfg, report=r.report)

    with st.expander("Modelled opponent range (not ground truth)"):
        st.caption("MODELLED OPPONENT RANGE — a Bayesian update of a "
                   "position/line prior under the selected behavior "
                   "archetype. It is a belief model, never the opponent's "
                   "actual holding.")
        model = st.selectbox("Behavior archetype",
                             ["regular", "nit", "calling_station", "maniac"],
                             key="hr_model")
        opp = next((s.seat for s in r.state.seats
                    if s.in_hand and s.seat != r.state.hero_seat), None)
        if opp is None:
            st.info("No live opponent.")
        else:
            stages = range_stages(r.state, opp, model=model)
            cols = st.columns(min(len(stages), 4))
            for i, stage in enumerate(stages):
                with cols[i % len(cols)]:
                    st.markdown(f"**{stage.name}**")
                    st.markdown(_range_matrix_html(stage),
                                unsafe_allow_html=True)
                    st.caption(f"{stage.entropy_bits:.1f} bits · "
                               f"≈{stage.effective_combos:.0f} effective / "
                               f"{stage.live_combos} live combos")
                    for w in stage.warnings:
                        st.caption("⚠ " + w)

    with st.expander("Decision path (how the solver saw it)"):
        for row in decision_path(r.report, r.state):
            line = f"**{html.escape(row.label)}** — {html.escape(row.value)}"
            st.markdown(line)
            if row.note:
                st.caption(row.note)


# ---------------------------------------------------------------------------
# Model Sensitivity
# ---------------------------------------------------------------------------

_VERDICT_BADGE = {"ROBUST": "high", "SENSITIVE": "medium",
                  "HIGHLY SENSITIVE": "low"}


def sensitivity_tab(cfg) -> None:
    from poker_alpha.demo_spots import DEMO_SPOTS

    st.caption("The same state analysed under each existing opponent "
               "archetype. No model is claimed correct — the question is "
               "whether the advice survives not knowing which one is. "
               "Rule: ROBUST = same top action everywhere and max "
               "total-variation < 0.20; HIGHLY SENSITIVE = two or more "
               "models disagree on the top action; SENSITIVE otherwise "
               "(or when the decision method itself changes).")
    name = st.selectbox("Spot", list(DEMO_SPOTS), key="ms_spot")
    if st.button("Run sensitivity analysis", key="ms_go", type="primary"):
        with st.spinner("Analysing under 4 archetypes…"):
            res = analyze_under_models(DEMO_SPOTS[name], config=cfg)
        st.session_state["ms_result"] = (name, res)
    stored = st.session_state.get("ms_result")
    if not stored or stored[0] != name:
        return
    res = stored[1]
    st.markdown(f"<span class='pa-badge "
                f"{_VERDICT_BADGE[res.verdict]}'>{res.verdict}</span> "
                f"<span class='pa-muted'>max distribution shift "
                f"{res.max_tv:.2f}</span>", unsafe_allow_html=True)
    st.dataframe([{
        "model": o.model, "top action": o.top_action or "—",
        "method": o.source_kind, "solver": o.solver_state,
        "confidence": o.confidence.upper(),
        "distribution": " / ".join(f"{l} {p:.0%}" for l, p in
                                   sorted(o.distribution.items(),
                                          key=lambda x: -x[1])[:4]),
    } for o in res.outcomes], hide_index=True)
    st.caption(res.note)


__all__ = ["replay_tab", "sensitivity_tab"]
