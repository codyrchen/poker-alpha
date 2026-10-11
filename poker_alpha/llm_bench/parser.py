"""Strict response parser. Illegal actions are NEVER silently coerced —
a response that names an illegal action, fails to parse, or carries a
non-normalized probability vector is recorded as a parse failure of the
appropriate category.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NORMALIZATION_TOL = 0.01


class ParseError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category   # no_json | bad_schema | illegal_action
        #                          | bad_probabilities


@dataclass
class ParsedResponse:
    action: str
    probabilities: Optional[Dict[str, float]] = None
    raw: str = field(default="", repr=False)


def parse_response(text: str, legal_actions: List[str]) -> ParsedResponse:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ParseError("no_json", "no JSON object found in response")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ParseError("no_json", f"invalid JSON: {exc}")
    if not isinstance(obj, dict) or "action" not in obj:
        raise ParseError("bad_schema", "missing 'action' field")
    action = obj["action"]
    if action not in legal_actions:
        raise ParseError("illegal_action",
                         f"{action!r} not in legal actions {legal_actions}")
    probs = obj.get("probabilities")
    if probs is not None:
        if not isinstance(probs, dict):
            raise ParseError("bad_probabilities", "probabilities not a dict")
        clean: Dict[str, float] = {}
        for k, v in probs.items():
            if k not in legal_actions:
                raise ParseError("bad_probabilities",
                                 f"probability for illegal action {k!r}")
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ParseError("bad_probabilities",
                                 f"non-numeric probability {v!r}")
            if f < -1e-9 or f > 1 + 1e-9:
                raise ParseError("bad_probabilities",
                                 f"probability {f} outside [0, 1]")
            clean[k] = min(max(f, 0.0), 1.0)
        total = sum(clean.values())
        if abs(total - 1.0) > NORMALIZATION_TOL:
            raise ParseError("bad_probabilities",
                             f"probabilities sum to {total:.4f}, not 1")
        # within tolerance: renormalize exactly (documented, not coercion
        # of an illegal response)
        probs = {k: v / total for k, v in clean.items()}
    return ParsedResponse(action=action, probabilities=probs, raw=text)


__all__ = ["ParseError", "ParsedResponse", "parse_response",
           "NORMALIZATION_TOL"]
