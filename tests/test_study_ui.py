"""Study Mode UI (headless): session review, explorer, what-if, trainer."""

from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "poker_alpha" / "ui" / "app.py"


def _study(timeout=600):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    at.sidebar.radio[0].set_value("Study").run()
    assert not at.exception, at.exception
    return at


@pytest.mark.slow
def test_study_session_review_and_matrix_headless():
    at = _study()
    # FLOW A: bundled session -> summary metrics
    labels = [m.label for m in at.metric]
    assert "Hands" in labels and "Decisions" in labels
    assert "Valid EV regret" in labels
    caps = " ".join(str(c.value) for c in at.caption)
    assert "never averaged in" in caps or "ESTIMATED" in caps
    # observer recognition never combined with solver trust
    assert "never combined" in caps
    # FLOW B: strategy matrix rendered from the artifact
    md = " ".join(str(m.value) for m in at.markdown)
    assert "border-collapse" in md          # the 13x13 grid
    assert "hard-coded" in caps             # provenance statement
    # exports stay local
    assert "nothing is" in caps and "uploaded" in caps


@pytest.mark.slow
def test_study_trainer_flow_headless():
    at = _study()
    btns = [b for b in at.button if str(b.key or "").startswith("tr_opt_")]
    assert btns, "trainer must offer the engine's real candidates"
    btns[0].click().run()
    assert not at.exception, at.exception
    md = " ".join(str(m.value) for m in at.markdown)
    assert "You chose" in md
    # grading language is never binary right/wrong
    assert "WRONG" not in md and "INCORRECT" not in md
    nxt = [b for b in at.button if b.key == "tr_next"]
    assert nxt
    nxt[0].click().run()
    assert not at.exception, at.exception


@pytest.mark.slow
def test_study_whatif_rejects_invalid_headless():
    at = _study()
    wi = [t for t in at.text_input if t.key == "wi_vals"][0]
    wi.set_value("3.75, -1").run()
    [b for b in at.button if b.key == "wi_go"][0].click().run()
    assert not at.exception, at.exception
    warn = " ".join(str(w.value) for w in at.warning)
    assert "rejected" in warn and "-1" in warn


@pytest.mark.slow
def test_play_and_developer_modes_still_intact():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception
    modes = at.sidebar.radio[0].options
    assert list(modes) == ["Play", "Study", "Developer"]
    md = " ".join(str(m.value) for m in at.markdown)
    assert "pa-action" in md                # Play demo still renders
    at.sidebar.radio[0].set_value("Developer").run()
    assert not at.exception, at.exception
    inputs = [r for r in at.sidebar.radio if r.label == "Input"]
    assert inputs and set(inputs[0].options) == {
        "Manual entry", "Hand-history replay", "Screen observer",
        "Live screen", "Annotate session"}
