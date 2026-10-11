"""Study Mode: session review, strategy explorer, what-if, trainer.

Everything here is driven by the same engine as Play Mode — the Study pages
only orchestrate :mod:`poker_alpha.analysis` and reuse the Play renderers
for any single decision, so the semantics (gate states, regret eligibility,
mixed-strategy language) cannot drift between surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from poker_alpha.analysis import (confidence_report, detect_patterns,
                                  review_hands, review_to_csv,
                                  review_to_json, review_to_markdown,
                                  summarize)
from poker_alpha.analysis.explorer import (PREFLOP_NODES, hand_classes,
                                           inspect_state, lookup_cell,
                                           matrix_stats, preflop_matrix)
from poker_alpha.analysis.trainer import (SPOT_LIBRARY, TrainerStats,
                                          answer_options, evaluate_answer,
                                          run_spot)
from poker_alpha.analysis.whatif import WHATIF_VARIABLES, compare_whatif
from poker_alpha.history import load_hands, parse_hands

from . import components as C
from .play import render_play_report
from .styles import ACCENT, CAUTION, MUTED

ROOT = Path(__file__).resolve().parents[2]
DEMO_SESSION = ROOT / "poker_alpha" / "data" / "demo_session.json"


# ---------------------------------------------------------------------------
# Session review
# ---------------------------------------------------------------------------

def _run_review(raw: bytes, cfg):
    key = (hash(raw), cfg.equity_simulations, cfg.rollout_simulations,
           cfg.seed, cfg.solver is not None)
    cached = st.session_state.get("study_review")
    if cached and cached[0] == key:
        return cached[1]
    hands = parse_hands(json.loads(raw))
    with st.spinner(f"Reviewing {len(hands)} hands through the engine…"):
        rev = review_hands(hands, config=cfg)
    st.session_state["study_review"] = (key, rev)
    return rev


def _review_tab(cfg) -> None:
    up = st.file_uploader("Session (pokeralpha.hand/v1 JSON)", type=["json"],
                          key="study_upload")
    if up is not None:
        raw = up.read()
    else:
        st.caption("Using the bundled demo session (8 heads-up hands). "
                   "Files you upload stay on this machine.")
        raw = DEMO_SESSION.read_bytes()
    try:
        rev = _run_review(raw, cfg)
    except (ValueError, KeyError) as exc:
        st.error(f"Cannot read session: {exc}")
        return
    s = summarize(rev.records, rev.hands, rev.warnings)
    pats = detect_patterns(rev.records)

    c = st.columns(6)
    c[0].metric("Hands", s.hands)
    c[1].metric("Decisions", s.decisions)
    c[2].metric("Solver accepted", s.solver_accepted)
    c[3].metric("Low confidence", s.solver_low_confidence)
    c[4].metric("Rejected", s.solver_rejected)
    c[5].metric("Valid EV regret", s.regret_eligible)
    if s.avg_estimated_regret_bb is not None:
        st.caption(f"Average ESTIMATED decision regret "
                   f"{s.avg_estimated_regret_bb:+.2f} BB over "
                   f"{s.regret_eligible} eligible decisions — a same-method "
                   f"rollout estimate, not money lost. Decisions without a "
                   f"valid comparison are never averaged in.")
    for w in s.warnings:
        st.warning(w)

    if pats:
        with st.expander(f"Patterns ({len(pats)})", expanded=False):
            for p in pats:
                st.markdown(f"**{p.label}** · {p.count}× — {p.description}  \n"
                            f"hands {', '.join(map(str, p.hands[:10]))}"
                            f"{'…' if len(p.hands) > 10 else ''} · "
                            f"{', '.join(f'{k}: {v}' for k, v in p.breakdown.items())}")

    rows = [{
        "hand": r.hand_index, "street": r.street, "pos": r.position or "—",
        "actual": r.actual_label or r.actual_kind,
        "PokerAlpha": r.recommended or "—",
        "freq": "" if r.actual_frequency is None else f"{r.actual_frequency:.0%}",
        "method": r.source_kind, "solver": r.solver_state,
        "conf": r.confidence,
        "est. regret": (f"{r.regret.estimated_regret:+.2f} BB"
                        if r.regret.eligible else "unavailable"),
        "category": r.category,
    } for r in rev.records]
    ev = st.dataframe(rows, hide_index=False, on_select="rerun",
                      selection_mode="single-row", key="study_table")
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        r = rev.records[sel[0]]
        st.markdown(f"#### Hand {r.hand_index}, {r.street} — you chose "
                    f"**{r.actual_label or r.actual_kind}**")
        st.caption(r.regret.display() + " · " + r.alignment.display())
        render_play_report(r.state, cfg, report=r.report)

    with st.expander("Session confidence report"):
        st.json(confidence_report(rev.records))
        st.caption("Observer recognition reliability (when present) and "
                   "solver trust are separate signals and are never combined.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Export JSON", review_to_json(rev, s, pats),
                       "pokeralpha_session_review.json", "application/json")
    d2.download_button("Export CSV", review_to_csv(rev),
                       "pokeralpha_session_review.csv", "text/csv")
    d3.download_button("Export Markdown", review_to_markdown(rev, s, pats),
                       "pokeralpha_session_review.md", "text/markdown")
    st.caption("Exports are written locally by your browser; nothing is "
               "uploaded anywhere.")


# ---------------------------------------------------------------------------
# Strategy explorer
# ---------------------------------------------------------------------------

_ACTION_COLOR = (("all_in", "#8a4fbf"), ("raise", ACCENT), ("bet", ACCENT),
                 ("call", "#2e9e5b"), ("check", "#2e9e5b"), ("fold", MUTED))


def _cell_color(label) -> str:
    for k, v in _ACTION_COLOR:
        if label and label.startswith(k):
            return v
    return MUTED


@st.cache_resource(show_spinner="Querying the strategy artifact…")
def _matrix(node: str, strategy_path: str):
    from poker_alpha.pipeline import load_solver

    return preflop_matrix(load_solver(strategy_path), node)


def _matrix_html(matrix) -> str:
    rows = []
    for row in matrix:
        tds = []
        for c in row:
            if c.status in ("ACCEPTED", "LOW CONFIDENCE"):
                col = _cell_color(c.dominant)
                op = "1.0" if c.status == "ACCEPTED" else "0.55"
                sub = f"{c.dominant_freq:.0%}" if c.dominant_freq else ""
                tds.append(
                    f'<td style="background:{col}22;border-left:3px solid '
                    f'{col};opacity:{op}"><b>{c.hand}</b><br>'
                    f'<span style="font-size:0.62rem">{c.dominant} {sub}</span></td>')
            else:
                tds.append(f'<td style="opacity:0.35"><b>{c.hand}</b><br>'
                           f'<span style="font-size:0.62rem">—</span></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return ('<div style="overflow-x:auto"><table style="border-collapse:'
            'separate;border-spacing:2px;font-size:0.72rem;line-height:1.15">'
            + "".join(rows) + "</table></div>")


def _explorer_tab(cfg, strategy_path: str) -> None:
    st.caption("Every frequency below is a live lookup against the promoted "
               "strategy artifact — nothing is hard-coded. Dimmed cells are "
               "states the confidence gate rejects or that the abstraction "
               "does not support; PokerAlpha shows no numbers there.")
    node = st.selectbox("Preflop node", list(PREFLOP_NODES), key="exp_node")
    matrix = _matrix(node, strategy_path)
    stats = matrix_stats(matrix)
    st.caption(" · ".join(f"{k}: {v}" for k, v in sorted(stats.items())))
    st.markdown(_matrix_html(matrix), unsafe_allow_html=True)

    flat = {c.hand: c for row in matrix for c in row}
    hand = st.selectbox("Inspect a hand class", list(flat), key="exp_hand")
    c = flat[hand]
    st.markdown(f"**{c.hand}** (queried as {c.combo}) — "
                f"<span class='pa-badge "
                f"{ {'ACCEPTED': 'high', 'LOW CONFIDENCE': 'medium'}.get(c.status, 'neutral') }'>"
                f"{c.status}</span>", unsafe_allow_html=True)
    if c.mix:
        st.dataframe([{"action": a, "frequency": f"{p:.1%}"} for a, p in c.mix],
                     hide_index=True)
        if c.visits is not None:
            st.caption(f"Training visits for this information set: {c.visits:,.0f}")
    for n in c.notes:
        st.caption("Note: " + n)
    if c.reasons:
        st.caption("Gate reasons: " + ", ".join(c.reasons))

    st.divider()
    st.markdown("**Postflop / arbitrary state** — paste any heads-up state "
                "to see how (and whether) the trained strategy answers it.")
    from poker_alpha.ui.play import DEMO_SPOTS

    default = json.dumps(DEMO_SPOTS["River, checked to hero — solver accepted"],
                         indent=1)
    text = st.text_area("State (pokeralpha.observed/v1 JSON)", default,
                        height=200, key="exp_state")
    if st.button("Inspect state", key="exp_go"):
        from poker_alpha.pipeline import load_solver

        try:
            r = inspect_state(load_solver(strategy_path), json.loads(text))
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        badge = {"ACCEPTED": "high", "LOW CONFIDENCE": "medium"}.get(r.status,
                                                                     "neutral")
        st.markdown(f"<span class='pa-badge {badge}'>{r.status}</span>",
                    unsafe_allow_html=True)
        if r.mix:
            st.dataframe([{"action": a, "frequency": f"{p:.1%}"}
                          for a, p in r.mix], hide_index=True)
        for n in r.notes:
            st.caption(n)
        if r.reasons:
            st.caption("Reasons: " + ", ".join(r.reasons))
        st.caption("The trained strategy covers specific heads-up trees; "
                   "UNSUPPORTED means this state is outside them — it does "
                   "not imply continuous GTO coverage anywhere.")


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------

def _whatif_tab(cfg) -> None:
    from poker_alpha.ui.play import DEMO_SPOTS

    st.caption("Change ONE variable at a time; each variant is re-validated "
               "and re-analysed by the real engine. Methods are compared by "
               "strategy and gate state — never by EV across different "
               "methods.")
    base_name = st.selectbox("Base spot", list(DEMO_SPOTS), key="wi_base")
    base = DEMO_SPOTS[base_name]
    with st.expander("Base state JSON"):
        st.code(json.dumps(base, indent=1), language="json")
    var = st.selectbox("Variable", WHATIF_VARIABLES, key="wi_var")
    hint = {"villain_bet": "bet sizes in bb, e.g. 1.65, 3.75, 7.5",
            "pot": "pot sizes in bb, e.g. 4, 8, 16",
            "effective_stack": "stacks in bb, e.g. 25, 50, 100",
            "hero_cards": "hands, e.g. Ah Kd, 9h 9c, 7s 6s (comma-separated)",
            "board_card": "cards, e.g. Kh, 2d, Qd"}[var]
    vals = st.text_input("Values (comma-separated)", key="wi_vals",
                         placeholder=hint)
    if st.button("Compare", key="wi_go") and vals.strip():
        values = [v.strip() for v in vals.split(",") if v.strip()]
        variants, rejected = compare_whatif(base, var, values, config=cfg)
        st.dataframe([{
            var: v.value, "solver": v.solver_state, "method": v.method,
            "confidence": v.confidence.upper(), "strategy": v.strategy_text(),
        } for v in variants], hide_index=True)
        for val, why in rejected.items():
            st.warning(f"{val}: rejected — {why}")
        st.caption("A rejected value describes an impossible state and was "
                   "not analysed.")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _trainer_tab(cfg) -> None:
    ss = st.session_state
    ss.setdefault("tr_i", 0)
    ss.setdefault("tr_feedback", None)
    if "tr_stats" not in ss:
        ss["tr_stats"] = TrainerStats()
    spot = SPOT_LIBRARY[ss["tr_i"] % len(SPOT_LIBRARY)]
    st.markdown(f"#### Spot {ss['tr_i'] % len(SPOT_LIBRARY) + 1} of "
                f"{len(SPOT_LIBRARY)} — {spot.name}")
    st.caption(spot.prompt)

    key = ("tr_report", ss["tr_i"], cfg.equity_simulations,
           cfg.rollout_simulations, cfg.seed)
    cached = ss.get("tr_report_cache")
    if cached and cached[0] == key:
        report = cached[1]
    else:
        with st.spinner("Preparing spot…"):
            report = run_spot(spot, config=cfg)
        ss["tr_report_cache"] = (key, report)

    from poker_alpha.pipeline import observe_manual
    from poker_alpha.ui.viewmodel import hand_vm

    C.render_hand(hand_vm(observe_manual(spot.state).state))

    if ss["tr_feedback"] is None:
        st.markdown("**Your action:**")
        cols = st.columns(min(len(answer_options(report)), 5) or 1)
        for i, opt in enumerate(answer_options(report)):
            if cols[i % len(cols)].button(opt, key=f"tr_opt_{opt}"):
                fb = evaluate_answer(report, opt)
                ss["tr_stats"].record(spot, fb)
                ss["tr_feedback"] = fb
                st.rerun()
    else:
        fb = ss["tr_feedback"]
        badge = {"COMMON": "high", "MIXED": "medium", "RARE": "medium",
                 "REGRET": "medium", "UNCERTAIN": "neutral"}.get(fb.grade,
                                                                 "neutral")
        st.markdown(f"You chose **{fb.chosen}** — "
                    f"<span class='pa-badge {badge}'>{fb.headline}</span>",
                    unsafe_allow_html=True)
        st.caption(fb.detail)
        with st.expander("Full PokerAlpha analysis", expanded=True):
            render_play_report(observe_manual(spot.state).state, cfg,
                               report=report)
        if st.button("Next spot", key="tr_next", type="primary"):
            ss["tr_i"] += 1
            ss["tr_feedback"] = None
            st.rerun()

    s = ss["tr_stats"].summary()
    if s["spots_attempted"]:
        st.divider()
        st.caption(
            f"This session: {s['spots_attempted']} spots · "
            + " · ".join(f"{k} {v}" for k, v in s["alignment"].items())
            + (f" · avg estimated regret "
               f"{s['avg_estimated_regret_bb']:+.2f} BB "
               f"(n={s['eligible_regret_n']})"
               if s["avg_estimated_regret_bb"] is not None else "")
            + " — stats stay in this browser session only.")


# ---------------------------------------------------------------------------

def study_mode(cfg, strategy_path: str) -> None:
    C.inject_styles()
    tabs = st.tabs(["Session Review", "Strategy Explorer", "What-if",
                    "Trainer"])
    with tabs[0]:
        _review_tab(cfg)
    with tabs[1]:
        _explorer_tab(cfg, strategy_path)
    with tabs[2]:
        _whatif_tab(cfg)
    with tabs[3]:
        _trainer_tab(cfg)


__all__ = ["study_mode"]
