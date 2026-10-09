"""Environment doctor: diagnose a PokerAlpha installation (Phase 74).

    python -m poker_alpha.doctor [--calibration FILE] [--monitor 1] [--no-capture] [--json]

Only DIAGNOSES: it never installs packages, changes system settings or
permissions, or creates folders. The screen-capture test grabs a 64x64
region in memory (never saved). Exit code 0 = no FAIL, 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str          # OK / WARN / FAIL / INFO
    detail: str
    hint: str = ""


def _version(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
    except Exception:  # noqa: BLE001 - any import problem = not usable
        return None
    return str(getattr(m, "__version__", "installed"))


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 11)
    return Check("python", "OK" if ok else "FAIL",
                 f"{platform.python_version()} ({sys.executable})",
                 "" if ok else "PokerAlpha needs Python >= 3.11")


def check_pokeralpha() -> Check:
    from .utils.provenance import git_commit

    try:
        from importlib.metadata import version
        ver = version("poker-alpha")
    except Exception:  # noqa: BLE001
        ver = "not installed as a package (running from source)"
    commit = git_commit()
    return Check("pokeralpha", "OK", f"version {ver}; commit {commit or 'unknown'}",
                 "" if commit else "not a Git checkout: commit unknown")


def check_strategy() -> Check:
    from .solver_config import RELEASE_CONFIG, RELEASE_STRATEGY
    from .solvers.strategy_artifact import StrategyArtifactError, load_artifact, manifest_path

    path = ROOT / RELEASE_STRATEGY
    if not path.exists():
        return Check("strategy artifact", "WARN", f"{path} missing",
                     "decisions fall back to rollouts / heuristics")
    try:
        art = load_artifact(path, RELEASE_CONFIG.build_game())
    except StrategyArtifactError as exc:
        return Check("strategy artifact", "FAIL", str(exc),
                     "restore the file from Git (git checkout -- results/strategy)")
    m = "manifest verified" if manifest_path(path).exists() else "no manifest"
    return Check("strategy artifact", "OK",
                 f"{RELEASE_STRATEGY}: {len(art)} infosets, {art.config_signature}, {m}")


def check_dependency(label: str, mod: str, required: bool, hint: str,
                     min_version: Optional[tuple] = None) -> Check:
    v = _version(mod)
    if v is None:
        return Check(label, "FAIL" if required else "WARN", "not installed", hint)
    if min_version and v != "installed":
        try:
            parts = tuple(int(x) for x in v.split(".")[:len(min_version)])
            if parts < min_version:
                return Check(label, "FAIL", f"{v} (< {'.'.join(map(str, min_version))})", hint)
        except ValueError:
            pass
    return Check(label, "OK", v)


def check_tesseract() -> Check:
    exe = shutil.which("tesseract")
    py = _version("pytesseract")
    if exe and py:
        return Check("tesseract OCR", "OK", f"{exe}; pytesseract {py}")
    return Check("tesseract OCR", "INFO",
                 f"binary {'found' if exe else 'not found'}, pytesseract "
                 f"{'installed' if py else 'not installed'}",
                 "optional: the built-in template OCR is used by default")


def check_monitors(monitor: int, capture: bool) -> List[Check]:
    try:
        from .observer.live import is_blank, list_monitors
    except ImportError as exc:
        return [Check("monitors", "FAIL", f"{type(exc).__name__}: {exc}",
                      "install the vision extra: pip install -e '.[vision]'")]
    try:
        mons = list_monitors()
    except ImportError as exc:
        return [Check("monitors", "FAIL", f"{type(exc).__name__}: {exc}",
                      "install the vision extra: pip install -e '.[vision]'")]
    except Exception as exc:  # noqa: BLE001 - no display / no permission
        return [Check("monitors", "FAIL", f"{type(exc).__name__}: {exc}",
                      "no screen is reachable: run PokerAlpha on the computer that shows "
                      "the table (not over SSH / in a container); on macOS grant Screen "
                      "Recording to the terminal app")]
    desc = ", ".join(f"#{m['index']} {m['width']}x{m['height']}@({m['left']},{m['top']})"
                     for m in mons)
    out = [Check("monitors", "OK" if len(mons) > 1 else "WARN", desc or "none",
                 "" if len(mons) > 1 else "no physical monitor reported")]
    if not capture:
        return out
    if monitor >= len(mons):
        out.append(Check("screen capture", "FAIL", f"monitor {monitor} does not exist"))
        return out
    try:
        from .observer.source import MSSScreenSource

        m = mons[monitor]
        img = MSSScreenSource((m["left"], m["top"], min(64, m["width"]),
                               min(64, m["height"]))).capture()
        if is_blank(img):
            out.append(Check("screen capture", "WARN",
                             f"monitor {monitor}: captured a uniform 64x64 patch",
                             "on macOS this usually means Screen Recording permission is "
                             "missing for the terminal app (System Settings -> Privacy & "
                             "Security -> Screen Recording), or the corner is a plain "
                             "background; PokerAlpha does not change permissions"))
        else:
            out.append(Check("screen capture", "OK",
                             f"monitor {monitor}: {img.size[0]}x{img.size[1]} px test patch "
                             "(in memory, not saved)"))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("screen capture", "FAIL", f"{type(exc).__name__}: {exc}",
                         "check Screen Recording permission for the terminal app"))
    return out


def check_calibration(path: Optional[str]) -> Check:
    from .observer.calibration import TableCalibration

    if not path:
        return Check("calibration", "INFO", "none given",
                     "the Live screen page starts from a preset (PokerNow Heads-Up or generic)")
    p = Path(path).expanduser()
    if not p.exists():
        return Check("calibration", "WARN", f"{p} not found")
    try:
        cal = TableCalibration.load(p)
    except Exception as exc:  # noqa: BLE001
        return Check("calibration", "FAIL", f"{p}: {exc}")
    return Check("calibration", "OK",
                 f"{cal.name}: {cal.num_seats} seats, detector {cal.table_detector}, "
                 f"client {cal.client}" + (f", migrated from {cal.migrated_from}"
                                           if cal.migrated_from else ""))


def _writable(folder: Path) -> Check:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    ok = os.access(probe, os.W_OK)
    state = "exists" if folder.exists() else f"will be created inside {probe}"
    return Check(f"write access {folder}", "OK" if ok else "FAIL", state,
                 "" if ok else "choose another folder in the UI")


def check_streamlit_privacy() -> Check:
    cfg = ROOT / ".streamlit" / "config.toml"
    if not cfg.exists():
        return Check("streamlit privacy config", "WARN", f"{cfg} missing",
                     "start Streamlit with --server.address 127.0.0.1 "
                     "--browser.gatherUsageStats false")
    text = cfg.read_text()
    ok = '127.0.0.1' in text and "gatherUsageStats = false" in text
    return Check("streamlit privacy config", "OK" if ok else "WARN",
                 f"{cfg} ({'local only, no usage stats' if ok else 'not the PokerAlpha defaults'})",
                 "" if ok else "see docs/privacy.md")


def run_checks(calibration=None, monitor=1, capture=True) -> List[Check]:
    checks: List[Check] = [check_python(), check_pokeralpha()]
    for label, mod, req, hint, mv in (
            ("numpy", "numpy", True, "pip install -e .", None),
            ("scipy", "scipy", True, "pip install -e .", None),
            ("Pillow", "PIL", True, "pip install -e '.[vision]'", None),
            ("mss", "mss", True, "pip install -e '.[vision]'", None),
            ("streamlit", "streamlit", True, "pip install -e '.[ui]'", (1, 37))):
        checks.append(check_dependency(label, mod, req, hint, mv))
    checks.append(check_tesseract())
    checks.append(check_strategy())
    checks += check_monitors(monitor, capture)
    checks.append(check_calibration(calibration))
    from .observer.session import DEFAULT_SESSION_ROOT

    checks.append(_writable(Path(DEFAULT_SESSION_ROOT).expanduser()))
    checks.append(_writable(Path("~/pokeralpha_captures").expanduser()))
    checks.append(check_streamlit_privacy())
    return checks


def main(argv=None, out: Callable[[str], None] = print) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calibration", default=None)
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--no-capture", action="store_true", help="skip the screen-capture test")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    checks = run_checks(a.calibration, a.monitor, not a.no_capture)
    if a.json:
        out(json.dumps([asdict(c) for c in checks], indent=1))
    else:
        out("PokerAlpha doctor (diagnose only; nothing is changed)")
        for c in checks:
            out(f"  [{c.status:<4}] {c.name}: {c.detail}")
            if c.hint and c.status != "OK":
                out(f"         -> {c.hint}")
        n_fail = sum(c.status == "FAIL" for c in checks)
        n_warn = sum(c.status == "WARN" for c in checks)
        out(f"{n_fail} FAIL, {n_warn} WARN")
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
