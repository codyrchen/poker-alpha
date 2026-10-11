"""External hand-history importers -> canonical ``pokeralpha.hand/v1``.

The canonical format stays canonical: importers are thin translators with
a tiny protocol, and anything the product consumes is canonical events
(:mod:`poker_alpha.history`), never an external format directly.

Protocol (module-level functions on each importer):

    name:            short format id
    detect(text)     -> bool: does this text look like the format?
    parse(text)      -> List[List[Event]] canonical hands (raises
                        ImportFormatError with a line/context message on
                        malformed input)

Dispatch helpers here:

    detect_format(text)            -> importer name or None
    parse_external_history(text, fmt=None) -> canonical hands

HONESTY RULE: an importer exists only for formats with representative
examples committed under ``tests/fixtures/imports/``. No support is
claimed for formats we have never seen. PokerNow CSV logs and
PokerStars/GGPoker text exports are FUTURE WORK pending real exported
samples — see docs/study_mode.md. This is post-session analysis only.
"""

from __future__ import annotations

from typing import List, Optional

from . import simple_lines

#: Registered importers, in detection order.
IMPORTERS = (simple_lines,)


class ImportFormatError(ValueError):
    """Malformed external history (carries a human-readable location)."""


def detect_format(text: str) -> Optional[str]:
    for imp in IMPORTERS:
        if imp.detect(text):
            return imp.name
    return None


def parse_external_history(text: str, fmt: Optional[str] = None):
    """Canonical hands from external text. ``fmt`` forces an importer."""
    if fmt is not None:
        for imp in IMPORTERS:
            if imp.name == fmt:
                return imp.parse(text)
        raise ImportFormatError(
            f"unknown format {fmt!r}; available: "
            f"{[i.name for i in IMPORTERS]}")
    name = detect_format(text)
    if name is None:
        raise ImportFormatError(
            "unrecognized hand-history format; available importers: "
            + ", ".join(i.name for i in IMPORTERS)
            + ". PokerNow/PokerStars exports are future work pending real "
              "sample files.")
    return parse_external_history(text, name)


__all__ = ["IMPORTERS", "ImportFormatError", "detect_format",
           "parse_external_history"]
