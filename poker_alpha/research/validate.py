"""Research-artifact validator (Phase II, Program 20G / 132).

Checks every ``results/phase2/*.json`` (and any paths given):
  * strict JSON: no NaN / Infinity / -Infinity tokens (Python's json writes
    them by default and most JSON parsers reject them);
  * a ``format`` string of the form ``pokeralpha.<...>/v<N>``;
  * ``commit`` (when present) is a 40-hex SHA;
  * no empty top-level collections named instances / runs / rows / results;
  * superseded artifacts carry an explicit ``superseded`` note.

    python -m poker_alpha.research.validate [paths...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
_FMT = re.compile(r"^pokeralpha\.[a-z0-9_.]+/v\d+$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def validate_file(path: Path) -> List[str]:
    problems = []
    try:
        doc = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    except ValueError as e:
        return [f"{path.name}: invalid strict JSON ({e})"]
    if not isinstance(doc, dict):
        return [f"{path.name}: top level is not an object"]
    fmt = doc.get("format")
    if not isinstance(fmt, str) or not _FMT.match(fmt):
        problems.append(f"{path.name}: missing/invalid format tag {fmt!r}")
    c = doc.get("commit")
    if c is not None and not (isinstance(c, str) and _SHA.match(c)):
        problems.append(f"{path.name}: commit is not a 40-hex SHA")
    for k in ("instances", "runs", "rows", "results"):
        if k in doc and isinstance(doc[k], (list, dict)) and len(doc[k]) == 0:
            problems.append(f"{path.name}: empty {k}")
    if "superseded" in path.stem and "superseded" not in doc:
        problems.append(f"{path.name}: superseded artifact without a 'superseded' note")
    return problems


def validate(paths=None) -> List[str]:
    files = [Path(p) for p in paths] if paths else sorted((ROOT / "results" / "phase2").glob("*.json"))
    out = []
    for f in files:
        out += validate_file(f)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    probs = validate(argv or None)
    n = len(argv) if argv else len(list((ROOT / "results" / "phase2").glob("*.json")))
    print("\n".join(probs) if probs else f"{n} artifacts valid")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
