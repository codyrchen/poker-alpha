"""Phases 74-75: environment doctor and live launcher (mocked screen)."""

import json
import subprocess
from pathlib import Path

import pytest

from poker_alpha import doctor, live

ROOT = Path(__file__).resolve().parents[1]


def _fake_monitors(monkeypatch, blank):
    pytest.importorskip("PIL")
    from PIL import Image

    import poker_alpha.observer.live as L
    import poker_alpha.observer.source as S
    mons = [{"index": 0, "left": 0, "top": 0, "width": 3024, "height": 1964},
            {"index": 1, "left": 0, "top": 0, "width": 1512, "height": 982}]
    monkeypatch.setattr(L, "list_monitors", lambda: mons)

    class Src:
        def __init__(self, box):
            self.box = box

        def capture(self):
            img = Image.new("RGB", (128, 128), (20, 20, 20))
            if not blank:
                img.paste((200, 50, 50), (0, 0, 64, 64))
            return img
    monkeypatch.setattr(S, "MSSScreenSource", Src)


def test_doctor_reports_capture_ok_and_permission_warning(monkeypatch):
    _fake_monitors(monkeypatch, blank=False)
    checks = {c.name: c for c in doctor.run_checks(monitor=1)}
    assert checks["monitors"].status == "OK" and checks["screen capture"].status == "OK"
    assert checks["strategy artifact"].status == "OK"
    _fake_monitors(monkeypatch, blank=True)
    checks = {c.name: c for c in doctor.run_checks(monitor=1)}
    assert checks["screen capture"].status == "WARN"
    assert "Screen Recording" in checks["screen capture"].hint


def test_doctor_json_exit_code_and_missing_monitor(monkeypatch, capsys):
    _fake_monitors(monkeypatch, blank=False)
    lines = []
    code = doctor.main(["--json", "--monitor", "5"], out=lines.append)
    data = json.loads(lines[0])
    assert any(c["name"] == "screen capture" and c["status"] == "FAIL" for c in data)
    assert code == 1


def test_doctor_never_creates_folders(tmp_path, monkeypatch):
    target = tmp_path / "a" / "b"
    c = doctor._writable(target)
    assert c.status == "OK" and not target.exists() and "will be created" in c.detail


def test_doctor_calibration_check(tmp_path):
    ok = doctor.check_calibration(str(ROOT / "tests/fixtures/pokernow/calibration.json"))
    assert ok.status == "OK"
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert doctor.check_calibration(str(bad)).status == "FAIL"


def test_launcher_command_is_local_and_quiet():
    cmd = live.streamlit_command(8600)
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"
    assert "8600" in cmd and cmd[-1] != "--force"


def test_launcher_stops_on_fail_and_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "main", lambda argv=None, out=print: 1)
    assert live.main(["--dry-run"]) == 1
    assert "Not starting" in capsys.readouterr().out
    assert live.main(["--dry-run", "--force"]) == 0
    assert "127.0.0.1" in capsys.readouterr().out


def test_shell_launcher_checks_python():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_live_observer.sh")])
    assert r.returncode == 0


def test_doctor_no_display_hint_is_not_an_install_hint(monkeypatch):
    pytest.importorskip("PIL")
    import poker_alpha.observer.live as L

    def no_display():
        raise RuntimeError("Cannot connect to display")
    monkeypatch.setattr(L, "list_monitors", no_display)
    c = {c.name: c for c in doctor.check_monitors(1, capture=False)}["monitors"]
    assert c.status == "FAIL"
    assert "no screen is reachable" in c.hint and "pip install" not in c.hint
