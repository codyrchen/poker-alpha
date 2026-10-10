"""Streamlit rendering of the Play Mode view models.

Thin: every function takes a view model from :mod:`poker_alpha.ui.viewmodel`
and emits markup. No solver access, no DecisionReport math.
"""

from __future__ import annotations

import html
from typing import Optional

import streamlit as st

from .styles import CSS
from .viewmodel import (AbstentionVM, HandVM, LiveStatusVM, RecommendationVM,
                        RefusalVM, WhyVM)


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _card_html(c, small: bool = False) -> str:
    cls = "pa-card" + (" red" if c.red else "") + (" small" if small else "")
    return f'<span class="{cls}">{html.escape(c.text)}</span>'


def _stat(k: str, v: str) -> str:
    return (f'<span class="pa-stat"><span class="v">{html.escape(v)}</span><br>'
            f'<span class="k">{html.escape(k)}</span></span>')


def _fmt_bb(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:,.1f} BB"


def render_hand(hand: HandVM, title: str = "") -> None:
    hero = "".join(_card_html(c) for c in hand.hero) or \
        '<span class="pa-muted">hero cards unknown</span>'
    board = "".join(_card_html(c, small=True) for c in hand.board) or \
        '<span class="pa-muted">no board yet</span>'
    pos = f" · {hand.position}" if hand.position else ""
    head = (f'<div class="pa-muted">{html.escape(title or hand.street)}'
            f'{html.escape(pos)}</div>')
    stats = "".join([
        _stat("Pot", _fmt_bb(hand.pot_bb)),
        _stat("To call", "—" if hand.to_call_bb is None else
              ("none" if hand.to_call_bb <= 0 else f"{hand.to_call_bb:g} BB")),
        _stat("Effective", _fmt_bb(hand.effective_bb)),
        _stat("SPR", "—" if hand.spr is None else f"{hand.spr:.1f}"),
    ])
    st.markdown(
        f'<div class="pa-panel">{head}'
        f'<div class="pa-cardrow">{hero}<span class="pa-sep">|</span>{board}</div>'
        f'<div class="pa-stats">{stats}</div></div>',
        unsafe_allow_html=True)


def render_actions(vm: RecommendationVM, max_actions: int = 6) -> None:
    rows = []
    for a in vm.actions[:max_actions]:
        pct = a.frequency_pct
        bar = 0 if pct is None else max(2, pct)
        ev = ""
        if a.ev_bb is not None:
            ev = f"{a.ev_bb:+.2f} BB"
            if a.ev_se_bb:
                ev += f" ±{a.ev_se_bb:.2f}"
        tag = '<div class="pa-rec-tag">RECOMMENDED</div>' if a.recommended else ""
        rows.append(
            f'<div class="pa-action{" rec" if a.recommended else ""}">'
            f'<span class="name">{html.escape(a.display)}{tag}</span>'
            f'<span class="bar"><div style="width:{bar}%"></div></span>'
            f'<span class="freq">{"—" if pct is None else f"{pct}%"}</span>'
            f'<span class="ev">{ev}</span></div>')
    st.markdown(f'<div class="pa-panel">{"".join(rows)}'
                f'<div class="pa-muted" style="margin-top:0.4rem">'
                f'{html.escape(vm.mix_meaning)}</div></div>',
                unsafe_allow_html=True)


_SOURCE_NAMES = {"solver": "Solver", "rollout": "Rollout", "heuristic": "Heuristic"}


def render_status(vm: RecommendationVM) -> None:
    badge = vm.confidence.lower() if vm.confidence.lower() in \
        ("high", "medium", "low") else "neutral"
    parts = [
        f'<span class="pa-stat"><span class="pa-badge {badge}">{vm.confidence}'
        f'</span><br><span class="k">Solver confidence</span></span>',
        _stat("Method", _SOURCE_NAMES.get(vm.source_kind, vm.method)),
    ]
    if vm.ev_edge_bb is not None:
        parts.append(_stat("EV edge", f"{vm.ev_edge_bb:+.2f} BB"))
    if vm.equity is not None:
        eq = f"{vm.equity:.0%}"
        if vm.equity_se:
            eq += f" ±{vm.equity_se:.0%}"
        parts.append(_stat("Equity", eq))
    if vm.pot_odds is not None:
        parts.append(_stat("Pot odds", f"{vm.pot_odds:.0%}"))
    st.markdown(f'<div class="pa-panel"><div class="pa-status">'
                f'{"".join(parts)}</div></div>', unsafe_allow_html=True)


def render_abstention(ab: AbstentionVM) -> None:
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in ab.reasons)
    fallback = ""
    if ab.withheld:
        fallback = ('<div class="pa-muted" style="margin-top:0.5rem">Fallback: '
                    + html.escape(ab.fallback or "no fallback available — no "
                                  "recommendation") + "</div>")
    st.markdown(
        f'<div class="pa-panel pa-abstain"><h4>'
        f'{"LOW CONFIDENCE" if not ab.withheld else "SOLVER WITHHELD"}</h4>'
        f'<div class="title">{html.escape(ab.title)}</div>'
        f'<div class="pa-muted">PokerAlpha does not trust the trained strategy '
        f'enough in this state.</div><ul>{reasons}</ul>{fallback}</div>',
        unsafe_allow_html=True)


def render_refusal(ref: RefusalVM) -> None:
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in ref.reasons[1:])
    st.markdown(
        f'<div class="pa-panel pa-wait">'
        f'<div class="title">{html.escape(ref.message)}</div>'
        f'<div class="pa-muted">No recommendation is shown for a state '
        f'PokerAlpha cannot trust.</div>'
        + (f"<ul>{reasons}</ul>" if reasons else "") + "</div>",
        unsafe_allow_html=True)


def render_live_status(vm: LiveStatusVM, solver_trust: Optional[str]) -> None:
    rec = "—" if vm.recognition is None else f"{vm.recognition:.0%}"
    badge = "neutral"
    if vm.recognition is not None:
        badge = "high" if vm.recognition >= 0.9 else \
            ("medium" if vm.recognition >= 0.5 else "error")
    parts = [
        f'<span class="pa-stat"><span class="pa-badge {badge}">{rec}</span><br>'
        f'<span class="k">Recognition</span></span>']
    if solver_trust:
        tb = solver_trust.lower() if solver_trust.lower() in \
            ("high", "medium", "low") else "neutral"
        parts.append(f'<span class="pa-stat"><span class="pa-badge {tb}">'
                     f'{html.escape(solver_trust)}</span><br>'
                     f'<span class="k">Solver trust</span></span>')
    st.markdown(f'<div class="pa-panel"><div class="pa-status">'
                f'{"".join(parts)}</div></div>', unsafe_allow_html=True)


def render_waiting(message: str) -> None:
    st.markdown(
        f'<div class="pa-panel pa-wait"><div class="title">'
        f'{html.escape(message)}</div>'
        f'<div class="pa-muted">PokerAlpha only analyses states it can read '
        f'reliably.</div></div>', unsafe_allow_html=True)


def render_why(why: WhyVM, report) -> None:
    """'Why this recommendation' expander; old tables live one level deeper."""
    from .view import (range_rows, solver_signal_rows, source_rows,
                       uncertainty_rows)

    with st.expander("Why this recommendation"):
        st.write(why.source_line)
        if why.gate_line:
            st.write(why.gate_line + (": " + "; ".join(why.gate_reasons)
                                      if why.gate_reasons else ""))
        if why.ev_line:
            st.write("EV: " + why.ev_line)
        for w in why.warnings:
            st.caption("⚠ " + w)
        def _str_rows(rows):
            return [{k: "" if v is None else str(v) for k, v in r.items()}
                    for r in rows]

        with st.expander("Technical details"):
            if solver_signal_rows(report):
                st.caption("Solver gate signals")
                st.dataframe(_str_rows(solver_signal_rows(report)), hide_index=True)
            st.caption("Decision sources (solver → rollout → heuristic)")
            st.dataframe(_str_rows(source_rows(report)), hide_index=True)
            st.caption("Uncertainty by source")
            st.dataframe(_str_rows(uncertainty_rows(report)), hide_index=True)
            if report.opponent_ranges:
                st.caption("Opponent ranges (beliefs, not known hands)")
                st.dataframe(_str_rows(range_rows(report)), hide_index=True)
