"""Phase 20: session storage and post-session analysis."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from poker_alpha.decision import DecisionConfig
from poker_alpha.history import load_hands, replay_hand
from poker_alpha.session import (SessionStore, actual_label, analyze_session,
                                 import_hands)

ROOT = Path(__file__).resolve().parents[1]
HANDS = ROOT / "tests" / "fixtures" / "hands" / "sample.json"
CFG = DecisionConfig(equity_simulations=300, rollout_simulations=200, seed=1)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    db = tmp_path_factory.mktemp("s") / "session.sqlite"
    store = SessionStore(db)
    sid = import_hands(store, load_hands(HANDS), config=CFG)
    yield store, sid, db
    store.close()


def test_everything_is_stored(stored):
    store, sid, db = stored
    assert store.sessions()[0]["id"] == sid
    hands = store.hands(sid)
    assert [h["hand_key"] for h in hands] == ["demo-1", "demo-2"]
    assert hands[0]["hero_net"] == pytest.approx(6.5)
    decs = store.decisions(sid)
    assert len(decs) == 5
    assert decs[0]["actual_kind"] == "raise" and decs[0]["report"] is not None
    assert decs[-1]["actual_label"] == "call"
    # Transparent format: plain SQL + JSON readable without PokerAlpha.
    con = sqlite3.connect(db)
    (n,) = con.execute("SELECT COUNT(*) FROM snapshots").fetchone()
    assert n > 0
    events = json.loads(con.execute("SELECT events_json FROM hands LIMIT 1").fetchone()[0])
    assert events[0]["type"] == "HandStarted"
    assert con.execute("SELECT value FROM meta").fetchone()[0] == "1"
    con.close()


def test_analysis_sections(stored):
    store, sid, _ = stored
    a = analyze_session(store, sid)
    assert a.hands == 2 and a.decisions == 5
    assert a.hero_net_bb == pytest.approx(-43.5)
    assert {"hero", "villain", "bb"} <= set(a.tendencies)
    assert a.tendencies["hero"]["pfr"].raw_successes == 2
    assert any(c.seat == 2 for c in a.range_changes)
    assert any(h == "demo-2" and "big pot" in r for h, r in a.review)
    for d in a.deviations:
        assert d.ev_loss_bb > 0
    text = a.format()
    assert "Hands worth reviewing" in text and "posterior" in text


def test_hands_without_analysis_and_schema_guard(tmp_path):
    db = tmp_path / "x.sqlite"
    with SessionStore(db) as store:
        sid = import_hands(store, load_hands(HANDS), analyze=False)
        assert all(d["report"] is None for d in store.decisions(sid))
        assert analyze_session(store, sid).deviations == ()
    con = sqlite3.connect(db)
    con.execute("UPDATE meta SET value='99'")
    con.commit()
    con.close()
    with pytest.raises(ValueError):
        SessionStore(db)


def test_actual_label_mapping():
    assert actual_label("fold", None, None) == "fold"
    assert actual_label("raise", 2.5, None) is None


def test_session_cli(tmp_path):
    db = tmp_path / "c.sqlite"
    out = subprocess.run([sys.executable, "-m", "poker_alpha.session", "import",
                          str(HANDS), "--db", str(db), "--rollouts", "100",
                          "--equity-sims", "200"], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    assert "imported session 1" in out
    out = subprocess.run([sys.executable, "-m", "poker_alpha.session", "analyze",
                          "--db", str(db)], capture_output=True, text=True,
                         check=True, cwd=ROOT).stdout
    assert "Session 1" in out
