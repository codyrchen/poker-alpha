"""Update results/validation/autonomous_progress.json (recovery state).

    python experiments/progress.py key=value [key=value ...]

Values are parsed as JSON when possible, else kept as strings. Nested keys
use dots: ``results.phase32=done``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "results" / "validation" / "autonomous_progress.json"


def update(**kv) -> dict:
    doc = json.loads(PATH.read_text()) if PATH.exists() else {}
    for k, v in kv.items():
        node = doc
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    try:
        doc["head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                              cwd=PATH.parents[2], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PATH.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


if __name__ == "__main__":
    kv = {}
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        try:
            kv[k] = json.loads(v)
        except ValueError:
            kv[k] = v
    print(json.dumps(update(**kv), indent=1))
