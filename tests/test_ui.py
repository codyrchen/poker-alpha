"""Phase 19: UI view helpers and a headless Streamlit smoke test."""

from pathlib import Path

import pytest

from poker_alpha.decision import DecisionConfig, recommend_action
from poker_alpha.holdem import ManualStateAdapter
from poker_alpha.ui.view import (candidate_rows, headline, range_rows,
                                 seat_rows, state_rows)

APP = Path(__file__).resolve().parents[1] / "poker_alpha" / "ui" / "app.py"


def _state():
    return ManualStateAdapter.from_dict({
        "num_seats": 3, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": "As Ks", "board": "Qs Js 4h",
        "pot": 13.5, "actor": 0,
        "seats": [{"stack": 96.5}, {"stack": 90.0, "bet": 4.5},
                  {"stack": 100.0, "folded": True}]})


def test_view_helpers():
    obs = _state()
    rows = {r["field"]: r["value"] for r in state_rows(obs)}
    assert rows["Hero"] == "As Ks" and rows["Board"] == "Qs Js 4h"
    assert rows["Amount to call"] == "4.5 BB" and rows["Position"] == "BTN"
    assert [r["status"] for r in seat_rows(obs)][0] == "HERO in hand"
    rep = recommend_action(obs, config=DecisionConfig(equity_simulations=200,
                                                      rollout_simulations=150))
    cands = candidate_rows(rep)
    assert {"action", "frequency", "EV (BB)", "± SE (BB)", "source"} <= set(cands[0])
    assert range_rows(rep)[0]["seat"] == 1
    assert "method" in headline(rep)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["Manual entry", "Hand-history replay",
                                  "Screen observer"])
def test_streamlit_app_runs_headless(mode):
    pytest.importorskip("streamlit")
    pytest.importorskip("PIL")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Developer").run()   # Play / Developer switch
    assert not at.exception, at.exception
    at.sidebar.slider[0].set_value(200)
    at.sidebar.slider[1].set_value(100)
    inputs = [r for r in at.sidebar.radio if r.label == "Input"][0]
    inputs.set_value(mode).run()
    assert not at.exception, at.exception
    if mode == "Screen observer":
        process = [b for b in at.button if b.label == "Process screenshot"][0]
        process.click().run()
        assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(
        str(s.value) for s in at.subheader)
    assert "PokerAlpha never" in " ".join(str(c.value) for c in at.caption)
    assert "Suggested" in text or "No recommendation" in text
