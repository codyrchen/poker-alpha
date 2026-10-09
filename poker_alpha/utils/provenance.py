"""Provenance helpers (no network, no subprocess)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit(repo: Optional[Path] = None) -> Optional[str]:
    """Commit of the checkout, read from .git files; None outside a checkout."""
    root = repo or REPO_ROOT
    try:
        ref = (root / ".git" / "HEAD").read_text().strip()
        if not ref.startswith("ref: "):
            return ref
        p = root / ".git" / ref[5:]
        if p.exists():
            return p.read_text().strip()
        for line in (root / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(ref[5:]):
                return line.split()[0]
    except OSError:
        return None
    return None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
