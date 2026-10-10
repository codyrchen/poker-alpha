"""Phase 47: per-region debug diagnostics (synthetic frames; explanation only)."""

import json

import pytest

pytest.importorskip("PIL")

from observer_helpers import CAL, MON, Screen, frames_of, table  # noqa: E402

from poker_alpha.observer.debug import (debug_report, region_rows,  # noqa: E402
                                        tracker_rows, write_debug_report)
from poker_alpha.observer.live import CaptureSettings, LiveObserverSession  # noqa: E402
from poker_alpha.observer.pokernow import FieldReading  # noqa: E402

pytestmark = pytest.mark.vision


def _session(frames):
    scr = Screen(frames)
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings(MON, None), CAL, 0.5, 1.0)
    return s


def _field(rows, name):
    return next(f for r in rows for f in r["fields"] if f["field"] == name)


def test_decisions_follow_card_confirmation():
    s = _session(frames_of(table(), repeat=3))
    s.step(now=0.0)
    rows = region_rows(s)
    assert {r["group"] for r in rows} == {"Hero", "Board", "Seats", "Pot/Bets", "Dealer/Actor"}
    hc = _field(rows, "hero_card_0")
    assert hc["raw"] == "As" and hc["decision"] == "pending" and "1/3" in hc["reason"]
    s.step(now=1.0)
    s.step(now=2.0)
    hc = _field(region_rows(s), "hero_card_0")
    assert hc["decision"] == "accepted" and hc["fused"] == "As"
    assert hc["previous_accepted"] is None
    s.step(now=3.0)
    assert _field(region_rows(s), "hero_card_0")["decision"] == "agrees"


def test_low_confidence_reading_is_reported_rejected():
    s = _session(frames_of(table(), repeat=1))
    s.step(now=0.0)
    orig = s.adapter.read_frame

    def low(img, timestamp=None):
        obs = orig(img, timestamp)
        obs.fields["pot"] = FieldReading(99.0, 0.05, "pot", timestamp)
        return obs
    s.adapter.read_frame = low
    s.step(now=1.0)
    pot = _field(region_rows(s), "pot")
    assert pot["decision"] == "rejected" and "below minimum" in pot["reason"]


def test_debug_report_does_not_change_tracker_and_writes_local_files(tmp_path):
    s = _session(frames_of(table(), repeat=3))
    for i in range(3):
        s.step(now=float(i))
    before = tracker_rows(s.tracker)
    rep = debug_report(s)
    assert tracker_rows(s.tracker) == before                 # explanation only
    assert rep["regions"] and rep["tracker"]
    paths = write_debug_report(s, tmp_path / "debug")
    assert (tmp_path / "debug" / ".gitignore").read_text().strip() == "*"
    data = json.loads(paths["json"].read_text())
    assert data["format"] == "pokeralpha.observer_debug_report/v1"
    html = paths["html"].read_text()
    assert "data:image/png;base64" in html and "never uploaded" in html
    p2 = write_debug_report(s, tmp_path / "debug")
    assert p2["json"] != paths["json"]                        # never overwritten
