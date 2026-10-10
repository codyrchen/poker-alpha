"""One-command live observer launcher (Phase 75).

    python -m poker_alpha.live [--port 8501] [--force] [--no-capture-test] [--dry-run]
    ./scripts/run_live_observer.sh              (same, from the repository)

1. runs the environment doctor (diagnose only);
2. on FAIL prints the problems with hints and stops (``--force`` starts anyway);
3. starts Streamlit on 127.0.0.1 with usage statistics off.

It never installs packages, changes permissions or edits system settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "poker_alpha" / "ui" / "app.py"


def streamlit_command(port: int = 8501) -> List[str]:
    return [sys.executable, "-m", "streamlit", "run", str(APP),
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--browser.gatherUsageStats", "false", "--browser.serverAddress", "localhost"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--force", action="store_true", help="start even if the doctor reports FAIL")
    p.add_argument("--no-capture-test", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the command, do not start")
    a = p.parse_args(argv)
    from .doctor import main as doctor

    code = doctor(["--no-capture"] if a.no_capture_test else [])
    cmd = streamlit_command(a.port)
    if code != 0 and not a.force:
        print("\nNot starting: fix the FAIL items above (or rerun with --force).")
        return code
    print("\nStarting (local only, usage statistics off):\n  " + " ".join(cmd))
    print("Open http://localhost:%d -> Input -> Live screen. Ctrl+C stops it." % a.port)
    if a.dry_run:
        return 0
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
