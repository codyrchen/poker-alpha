"""Streamlit page: annotate frames of a recorded observer test session.

Local only: reads the session folder, writes ``annotations/<id>.json``.
Logic lives in :mod:`poker_alpha.observer.annotation_tool`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..observer.annotation_tool import (SEAT_STATUSES, SessionAnnotator, build_annotation,
                                        seat_fields_to_status)
from ..observer.annotations import ROLES, STATUSES
from ..observer.session import DEFAULT_SESSION_ROOT

STREET_CHOICES = ("unknown", "preflop", "flop", "turn", "river")


def _sessions(root: Path):
    if (root / "session.json").exists():
        return [root]
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if (p / "session.json").exists()), reverse=True)


def _fmt(v) -> str:
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def annotate_mode() -> None:
    st.subheader("Annotate a recorded observer session")
    st.caption("Ground truth for saved frames. Blank = unknown (not scored). Local files "
               "only. Prefilled values are the observer's own readings: check every one.")
    root = Path(st.text_input("Sessions folder (or one session)", DEFAULT_SESSION_ROOT,
                              key="ann_root")).expanduser()
    sessions = _sessions(root)
    if not sessions:
        st.info(f"No recorded sessions under {root}. Record one in Live screen -> "
                "Observer Test Session.")
        return
    path = st.selectbox("Session", sessions, format_func=lambda p: p.name, key="ann_session")
    cache = st.session_state.get("ann_tool")
    if cache is None or cache.path != Path(path):
        try:
            cache = SessionAnnotator(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cannot open session: {exc}")
            return
        st.session_state.ann_tool = cache
        st.session_state.ann_idx = 0
    tool: SessionAnnotator = cache
    ids = tool.ids
    if not ids:
        st.info("This session has no saved frames.")
        return
    prog = tool.progress()
    st.markdown(f"**Annotated {prog['annotated']} / {prog['total']}** · complete "
                f"{prog['complete']} · partial {prog['partial']} · skipped {prog['skip']} · "
                f"unreviewed {prog['unreviewed']}")
    idx = min(st.session_state.get("ann_idx", 0), len(ids) - 1)
    c = st.columns([1, 1, 2, 3])
    if c[0].button("◀ Previous", disabled=idx == 0, key="ann_prev"):
        st.session_state.ann_idx = idx - 1
        st.rerun()
    if c[1].button("Next ▶", disabled=idx >= len(ids) - 1, key="ann_next"):
        st.session_state.ann_idx = idx + 1
        st.rerun()
    if c[2].button("Next unreviewed", key="ann_next_unrev"):
        nxt = tool.next_unreviewed(ids[idx])
        if nxt is not None:
            st.session_state.ann_idx = ids.index(nxt)
            st.rerun()
        st.info("No unreviewed frames left.")
    pick = c[3].selectbox("Frame", ids, index=idx, key=f"ann_pick_{idx}",
                          format_func=lambda i: f"{i} · {tool.status(i)}")
    if pick != ids[idx]:
        st.session_state.ann_idx = ids.index(pick)
        st.rerun()
    sid = ids[idx]
    readings = tool.observer_readings(sid)
    left, right = st.columns([3, 2])
    with left:
        st.image(str(tool.frame_path(sid)), caption=f"frame {sid}")
    with right:
        diag = readings.get("diagnostics", {})
        st.caption(f"kept because: {', '.join(diag.get('reasons', [])) or 'n/a'}")
        st_ = readings.get("tracked_state")
        if st_:
            snap = st_["snapshot"]
            st.markdown("**Observer (fused) — for reference only**")
            st.json({"hero_cards": snap["hero_cards"], "board": snap["board"],
                     "pot": snap["pot"], "dealer": snap["dealer"], "actor": snap["actor"],
                     "stacks": snap["stacks"], "bets": snap["bets"],
                     "in_hand": snap["in_hand"], "occupied": snap["occupied"]},
                     expanded=False)
        if st.button("Prefill from observer readings", key=f"ann_prefill_{sid}"):
            st.session_state[f"ann_draft_{sid}"] = tool.prefill(sid)
            st.rerun()
    saved = tool.load(sid)
    draft = saved or st.session_state.get(f"ann_draft_{sid}") or tool.empty(sid)
    if draft.get("prefilled_from_observer") and not saved:
        st.warning("Prefilled from the observer: verify every field before saving.")
    n = int(draft["num_seats"])
    seats = {s["seat"]: s for s in draft.get("seats", [])}
    with st.form(f"ann_form_{sid}"):
        c = st.columns(4)
        hc = draft.get("hero_cards") or ["", ""]
        h1 = c[0].text_input("Hero card 1", hc[0] if len(hc) > 0 else "", key=f"a_h1_{sid}")
        h2 = c[1].text_input("Hero card 2", hc[1] if len(hc) > 1 else "", key=f"a_h2_{sid}")
        board = c[2].text_input("Board ('-' = none)", " ".join(draft["board"])
                                if draft.get("board") is not None and draft.get("board")
                                else ("-" if draft.get("board") == [] else ""),
                                key=f"a_board_{sid}")
        street = c[3].selectbox("Street", STREET_CHOICES,
                                index=STREET_CHOICES.index(draft.get("street", "unknown")
                                                           or "unknown"),
                                key=f"a_street_{sid}")
        seat_opts = ["unknown", "none"] + [str(i) for i in range(n)]

        def _seat_idx(key):
            if key not in draft:
                return 0
            return 1 if draft[key] is None else seat_opts.index(str(draft[key]))
        c = st.columns(4)
        dealer = c[0].selectbox("Dealer seat", seat_opts, index=_seat_idx("dealer_seat"),
                                key=f"a_dealer_{sid}")
        actor = c[1].selectbox("Actor seat", seat_opts, index=_seat_idx("actor"),
                               key=f"a_actor_{sid}")
        pot = c[2].text_input("Pot (as displayed)", _fmt(draft.get("pot")), key=f"a_pot_{sid}")
        pot_total = c[3].text_input("Total pot (if known)", _fmt(draft.get("pot_total")),
                                    key=f"a_pott_{sid}")
        st.markdown(f"**Seats** (hero = seat {draft['hero_seat']})")
        seat_rows = []
        for i in range(n):
            s = seats.get(i, {})
            c = st.columns([1, 2, 2, 2])
            c[0].markdown(f"seat {i}{' (hero)' if i == draft['hero_seat'] else ''}")
            stat = c[1].selectbox("status", SEAT_STATUSES,
                                  index=SEAT_STATUSES.index(seat_fields_to_status(s)),
                                  key=f"a_s{i}_{sid}", label_visibility="collapsed")
            stack = c[2].text_input("stack", _fmt(s.get("stack")), key=f"a_st{i}_{sid}",
                                    placeholder="stack")
            bet = c[3].text_input("bet", _fmt(s.get("bet")), key=f"a_b{i}_{sid}",
                                  placeholder="bet")
            seat_rows.append({"status": stat, "stack": stack, "bet": bet})
        notes = st.text_area("Notes", draft.get("notes", ""), key=f"a_notes_{sid}")
        c = st.columns(2)
        status = c[0].selectbox("Status", STATUSES,
                                index=STATUSES.index(draft.get("status", "unreviewed")
                                                     if saved else "complete"),
                                key=f"a_status_{sid}")
        role = c[1].selectbox("Dataset role", ROLES,
                              index=ROLES.index(draft.get("role", "unassigned")),
                              key=f"a_role_{sid}",
                              help="tuning = used to fit recognizers; validation / "
                                   "held_out = never used for fitting")
        b = st.columns(2)
        save = b[0].form_submit_button("Save")
        save_next = b[1].form_submit_button("Save & next")
    if save or save_next:
        try:
            form = build_annotation(tool.empty(sid), status=status, role=role,
                                    hero_cards=(h1, h2), board=board, street=street,
                                    dealer=dealer, actor=actor, pot=pot,
                                    pot_total=pot_total, seats=seat_rows, notes=notes)
            tool.save(sid, form)
            st.session_state.pop(f"ann_draft_{sid}", None)
            st.success(f"Saved annotations/{sid}.json ({status})")
            if save_next and idx < len(ids) - 1:
                st.session_state.ann_idx = idx + 1
                st.rerun()
        except ValueError as exc:
            st.error(f"Not saved: {exc}")
