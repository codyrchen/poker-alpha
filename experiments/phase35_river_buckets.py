"""Phase 35: does an exact river-strength percentile bucket reduce the
compact encoder's measured abstraction error? Same subgames and method as
Phase 34A (exact CFR+ per encoder, evaluated in the raw game).

Writes results/validation/abstraction_error_river_pct.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase34_abstraction_error import BOARDS, part_a  # noqa: E402

if __name__ == "__main__":
    rows = part_a(BOARDS, 400, ("raw", "compact", "compact_river_pct10", "compact_river_pct20"))
    out = ROOT / "results" / "validation" / "abstraction_error_river_pct.json"
    out.write_text(json.dumps({"format": "pokeralpha.abstraction_error_river_pct/v1", "rows": rows},
                              indent=1, default=float))
    print("wrote", out)
