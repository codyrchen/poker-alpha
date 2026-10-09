"""Phase 42: deterministic replay of recorded observer sessions (synthetic)."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("PIL")

from observer_helpers import CAL, frames_of, record, table  # noqa: E402

from poker_alpha.observer.live import transform_regions  # noqa: E402
from poker_alpha.observer.session import RetentionPolicy  # noqa: E402
from poker_alpha.observer.session_replay import (compare_reports, load_session,  # noqa: E402
                                                 main, replay)

pytestmark = pytest.mark.vision

TIMELINE = (table(), table(board=("Qs", "Jh", "4c"), pot=3.0),
            table(board=("Qs", "Jh", "4c"), pot=3.0, bet1=2.0, stacks=(99.0, 96.0)),
            table(board=("Qs", "Jh", "4c", "2d"), pot=7.0),
            table(dealer=0, hero=("7c", "7d"), pot=1.5))


def _reset(s, rec):
    s.reset_tracker()


def _recal(s, rec):
    c = replace(CAL, name="retuned", felt_tolerance=CAL.felt_tolerance + 1)   # new geometry
    s.configure(s.capture, c, 0.5, 1.0)


@pytest.fixture
def session_dir(tmp_path):
    rec, s = record(tmp_path, frames_of(*TIMELINE, repeat=4), actions={9: _reset})
    rec.finish()
    return rec.path


def test_stored_replay_reproduces_live_tracker_exactly(session_dir):
    rep = replay(session_dir, "stored")
    assert rep["reproduced"], json.dumps(rep, indent=1)[:2000]
    assert rep["stream_frames"] == 20 and rep["samples"] >= 5
    assert rep["events_live"] == rep["events_replayed"] > 0


def test_stored_replay_follows_recalibration(tmp_path):
    rec, s = record(tmp_path, frames_of(*TIMELINE[:3], repeat=4), actions={6: _recal})
    rec.finish()
    data = load_session(rec.path)
    assert len(data.calibrations) == 2                     # initial + changed
    assert sum(1 for r in data.stream if r.get("reset")) == 2
    assert replay(rec.path, "stored")["reproduced"]


def test_stored_replay_detects_a_changed_tracker(session_dir, monkeypatch):
    from poker_alpha.observer import fusion

    orig = fusion.TrackerConfig
    monkeypatch.setattr(fusion, "TrackerConfig", lambda: orig(card_confirm_frames=1))
    rep = replay(session_dir, "stored")
    assert not rep["reproduced"] and rep["samples_differing"] > 0


def test_recompute_with_same_code_changes_nothing(session_dir):
    rep = replay(session_dir, "recompute", decisions=True, sims=100)
    assert rep["samples_with_value_changes"] == 0
    assert not rep["tracker_differences"] and not rep["decision_differences"]
    assert all(p["status"] == "match" for p in rep["per_sample"])


def test_compare_calibration_reports_field_changes(session_dir, tmp_path):
    shifted = transform_regions(CAL, dx=0.03, dy=0.02)
    shifted.save(tmp_path / "shifted.json")
    rep = replay(session_dir, "recompute", compare_calibration=shifted, decisions=False)
    assert rep["baseline"].startswith("recompute with session")
    assert rep["samples_with_value_changes"] > 0
    changed = {d["field"] for p in rep["per_sample"] for d in p.get("field_differences", [])
               if d["value_changed"]}
    assert changed


def test_cli_reports_and_cross_report_comparison(session_dir, tmp_path, capsys):
    a = tmp_path / "a.json"
    assert main([str(session_dir), "--mode", "recompute", "--no-decisions",
                 "--write-report", str(a)]) == 0
    old = json.loads(a.read_text())
    assert compare_reports(old, old)["samples_differing"] == 0
    tampered = json.loads(a.read_text())
    k = sorted(tampered["candidate_results"])[-1]
    tampered["candidate_results"][k]["state"]["pot"] = 999.0
    (tmp_path / "t.json").write_text(json.dumps(tampered))
    assert main([str(session_dir), "--mode", "recompute", "--no-decisions",
                 "--compare-report", str(tmp_path / "t.json")]) == 0
    out = capsys.readouterr().out
    assert "1 of" in out and "samples differ" in out
    assert main([str(session_dir)]) == 0
    assert "reproduced: True" in capsys.readouterr().out


def test_crashed_session_without_manifest_json_still_replays(tmp_path):
    rec, s = record(tmp_path, frames_of(*TIMELINE[:2], repeat=4))
    # no finish(): manifest.json was never written, only the streamed manifest.jsonl
    assert not (rec.path / "manifest.json").exists()
    with open(rec.path / "observations" / "stream.jsonl", "a") as fh:
        fh.write('{"n": 99, "ok": tr')                  # truncated last line
    rep = replay(rec.path, "stored")
    assert rep["samples"] == len(rec.samples) and rep["reproduced"]


def test_manual_only_session_replays(tmp_path):
    rec, s = record(tmp_path, frames_of(*TIMELINE[:2], repeat=3), policy=RetentionPolicy(()))
    rec.mark(s, "only this")
    rec.finish()
    rep = replay(rec.path, "recompute", decisions=False)
    assert rep["samples"] == 1 and rep["samples_with_value_changes"] == 0
