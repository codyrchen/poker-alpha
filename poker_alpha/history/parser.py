"""The PokerAlpha JSON hand-history format (``pokeralpha.hand/v1``).

::

    {"format": "pokeralpha.hand/v1",
     "hands": [{"events": [{"type": "HandStarted", ...}, ...]}, ...]}

A single-hand file may use ``{"format": ..., "events": [...]}``. Events use
the field names of :mod:`poker_alpha.history.events`. Source-specific
formats (e.g. a PokerNow export) belong in their own adapter modules that
emit these events; the core never parses site formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .events import Event, event_from_dict

HAND_FORMAT = "pokeralpha.hand/v1"


def parse_hands(data: Dict[str, Any]) -> List[List[Event]]:
    fmt = data.get("format")
    if fmt != HAND_FORMAT:
        raise ValueError(f"unsupported hand-history format {fmt!r} "
                         f"(expected {HAND_FORMAT})")
    hands = data["hands"] if "hands" in data else [{"events": data["events"]}]
    return [[event_from_dict(e) for e in h["events"]] for h in hands]


def load_hands(path: Union[str, Path]) -> List[List[Event]]:
    return parse_hands(json.loads(Path(path).read_text(encoding="utf-8")))


def dump_hands(hands: Sequence[Sequence[Event]]) -> Dict[str, Any]:
    return {"format": HAND_FORMAT,
            "hands": [{"events": [e.to_dict() for e in h]} for h in hands]}


def save_hands(hands: Sequence[Sequence[Event]], path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(dump_hands(hands), indent=1),
                          encoding="utf-8")
