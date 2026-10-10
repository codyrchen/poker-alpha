"""Play Mode view models and page (product UI v0.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from poker_alpha.decision.report import CandidateAction, DecisionReport
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.ui.viewmodel import (compact_summary, hand_vm, live_status_vm,
                                      reason_text, recommendation_vm,
                                      waiting_message, why_vm)

APP = Path(__file__).resolve().parents[1] / "poker_alpha" / "ui" / "app.py"


def _report(candidates, recommended, method="Monte Carlo rollout",
            confidence="medium", solver_info=None, warnings=(),
            equity=0.62, equity_se=0.01, refusal=None):
    details = {"solver": solver_info or
               {"used": False, "confidence": "not configured", "reasons": []},
               "source_cascade": []}
    if refusal:
        details["refusal_codes"] = refusal
    return DecisionReport(
        state_summary="s", hero_equity=equity, hero_equity_se=equity_se,
        pot_bb=10.0, to_call_bb=0.0, pot_odds=0.25, spr=4.0,
        effective_stack_bb=40.0, opponent_ranges=(),
        candidates=tuple(candidates), recommended=recommended,
        recommended_mix={}, mix_meaning="mixed strategy frequencies",
        method=method, confidence=confidence, warnings=tuple(warnings),
        details=details, uncertainty={"sampling": "±0.1"})


def _cand(label, kind, prob, ev=None, se=None, to=0.0, added=0.0,
          source="Monte Carlo rollout"):
    return CandidateAction(label=label, kind=kind, amount_to=to, added=added,
                           probability=prob, ev_bb=ev, ev_se_bb=se,
                           source=source)


def test_actions_sorted_by_frequency_and_recommended_flag():
    rep = _report([_cand("check", "check", 0.18),
                   _cand("bet_75", "bet", 0.64, ev=1.2, se=0.1, added=6.2),
                   _cand("bet_150", "bet", 0.18, ev=0.9, se=0.1, added=9.4)],
                  recommended="bet_75")
    vm = recommendation_vm(rep)
    assert [a.label for a in vm.actions] == ["bet_75", "check", "bet_150"]
    assert vm.recommended.label == "bet_75"
    assert vm.recommended.display == "Bet 6.2 BB"
    assert vm.actions[0].frequency_pct == 64


def test_ev_edge_and_missing_ev():
    rep = _report([_cand("bet_75", "bet", 0.6, ev=1.2, added=6.0),
                   _cand("check", "check", 0.4, ev=0.9)], "bet_75")
    assert recommendation_vm(rep).ev_edge_bb == pytest.approx(0.3)
    # solver-only candidates carry no EVs -> edge is None, nothing invented
    rep2 = _report([_cand("bet_75", "bet", 0.6, source="solver"),
                    _cand("check", "check", 0.4, source="solver")], "bet_75",
                   method="solver")
    vm2 = recommendation_vm(rep2)
    assert vm2.ev_edge_bb is None and vm2.recommended.ev_bb is None


def test_missing_equity_handled():
    rep = _report([_cand("check", "check", 1.0)], "check",
                  equity=None, equity_se=None)
    vm = recommendation_vm(rep)
    assert vm.equity is None


def test_confidence_labels_and_sources():
    for conf, label in (("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH")):
        rep = _report([_cand("check", "check", 1.0)], "check", confidence=conf)
        assert recommendation_vm(rep).confidence == label
    for method, kind in (("solver", "solver"),
                         ("interpolated abstraction", "solver"),
                         ("Monte Carlo rollout", "rollout"),
                         ("heuristic fallback", "heuristic")):
        rep = _report([_cand("check", "check", 1.0)], "check", method=method)
        assert recommendation_vm(rep).source_kind == kind


def test_abstention_from_rejected_gate():
    rep = _report([_cand("call", "call", 1.0, ev=0.5, added=2.0)], "call",
                  method="Monte Carlo rollout",
                  solver_info={"used": False, "confidence": "rejected",
                               "reasons": ["HIGH_SEED_DISAGREEMENT",
                                           "LOW_VISIT_COUNT"]})
    vm = recommendation_vm(rep)
    ab = vm.abstention
    assert ab is not None and ab.withheld
    assert "independently trained runs disagree here" in ab.reasons
    assert ab.fallback == "Monte Carlo rollout"
    assert ab.reason_codes == ("HIGH_SEED_DISAGREEMENT", "LOW_VISIT_COUNT")


def test_low_confidence_downgrade_not_withheld():
    rep = _report([_cand("call", "call", 1.0, source="solver")], "call",
                  method="solver",
                  solver_info={"used": True,
                               "confidence": "SOLVER_LOW_CONFIDENCE",
                               "reasons": ["STREET_ABSTRACTION_ERROR"]})
    ab = recommendation_vm(rep).abstention
    assert ab is not None and not ab.withheld
    assert "abstraction error" in ab.reasons[0]


def test_refusal_state():
    rep = _report([], None, method="none", confidence="low",
                  refusal=["NOT_HERO_TURN"])
    vm = recommendation_vm(rep)
    assert vm.refusal is not None
    assert vm.refusal.message == "It is not the hero's turn"
    cs = compact_summary(rep)
    assert cs.withheld and cs.lines == ()


def test_compact_overlay_summary():
    rep = _report([_cand("bet_75", "bet", 0.64, added=6.2),
                   _cand("check", "check", 0.18),
                   _cand("bet_150", "bet", 0.18, added=9.4),
                   _cand("fold", "fold", 0.0)], "bet_75", confidence="high")
    cs = compact_summary(rep, max_lines=3)
    assert cs.lines == (("Bet 6.2 BB", "64%"), ("Check", "18%"),
                        ("Bet 9.4 BB", "18%"))
    assert cs.confidence == "HIGH CONFIDENCE"
    assert "Bet 6.2 BB" in cs.text() and "HIGH" in cs.text()


def test_hand_vm_summary():
    obs = ManualStateAdapter.from_dict({
        "num_seats": 3, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": "As Kd", "board": "Qs Js 4h",
        "pot": 13.5, "actor": 0,
        "seats": [{"stack": 96.5}, {"stack": 90.0, "bet": 4.5},
                  {"stack": 100.0, "folded": True}]})
    hv = hand_vm(obs)
    assert [c.text for c in hv.hero] == ["A♠", "K♦"]
    assert hv.hero[1].red and not hv.hero[0].red
    assert len(hv.board) == 3 and hv.street == "Flop"
    assert hv.position == "BTN"
    assert hv.pot_bb == 13.5 and hv.to_call_bb == 4.5
    assert "to call" in hv.facing


def test_waiting_messages_and_reason_text():
    assert waiting_message(["hero cards unconfirmed"]).startswith(
        "Waiting for a reliable hero-card")
    assert waiting_message(["pot read invalid"]) == \
        "Waiting for a reliable pot read"
    assert waiting_message(["???"]) == "Waiting for a reliable table read"
    st = live_status_vm(False, 0.4, ["pot broken"], paused=False)
    assert st.waiting_for == "Waiting for a reliable pot read"
    assert live_status_vm(True, 0.9, paused=True).waiting_for == "Observer paused"
    assert reason_text("HIGH_SEED_DISAGREEMENT") == \
        "independently trained runs disagree here"
    assert reason_text("SOME_NEW_CODE") == "some new code"


def test_why_vm_lines():
    rep = _report([_cand("bet_75", "bet", 0.6, ev=1.2, added=6.0),
                   _cand("check", "check", 0.4, ev=0.9)], "bet_75",
                  method="Monte Carlo rollout",
                  warnings=("observer confidence 80%",),
                  solver_info={"used": False, "confidence": "rejected",
                               "reasons": ["UNSEEN_STATE"]})
    w = why_vm(rep)
    assert "rollout" in w.source_line.lower()
    assert w.gate_line == "Confidence gate: rejected"
    assert "never came up in training" in w.gate_reasons[0]
    assert "+0.30 BB vs the next-best action" in w.ev_line
    assert w.warnings == ("observer confidence 80%",)


# ---------------------------------------------------------------------------
# end-to-end: the real pipeline through the Play page (headless)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_play_mode_demo_and_abstention_headless():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    at.sidebar.slider[0].set_value(200)
    at.sidebar.slider[1].set_value(150)
    at.run()
    assert not at.exception, at.exception
    md = " ".join(str(m.value) for m in at.markdown)
    assert "pa-action" in md and "RECOMMENDED" in md        # action rows
    assert "pa-card" in md                                   # card chips
    sel = [s for s in at.selectbox if s.key == "play_demo_spot"][0]
    sel.set_value("Preflop open decision — gate rejects (seed disagreement)").run()
    assert not at.exception, at.exception
    md = " ".join(str(m.value) for m in at.markdown)
    assert "Solver recommendation withheld" in md
    assert "independently trained runs disagree here" in md
    assert "Fallback" in md


@pytest.mark.slow
def test_developer_mode_preserved_headless():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Developer").run()
    assert not at.exception, at.exception
    labels = [r.label for r in at.sidebar.radio]
    assert "Input" in labels
    inputs = [r for r in at.sidebar.radio if r.label == "Input"][0]
    assert set(inputs.options) == {"Manual entry", "Hand-history replay",
                                   "Screen observer", "Live screen",
                                   "Annotate session"}
