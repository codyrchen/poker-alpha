"""Solver-gate acceptance by street (Phase 64).

For every key of a confidence table, apply the gate with the calibrated
thresholds and report, per street, the share of keys and the visit-weighted
share (a proxy for how often the spot occurs in self-play) that is accepted,
low-confidence or rejected, with the reason counts.

    python experiments/gate_acceptance.py results/strategy/holdem_v2_seed0_confidence.npz \
        --out results/validation/gate_acceptance_v2_100k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision.solver_gate import (ACCEPT, LOW, REJECT, ConfidenceTable,  # noqa: E402
                                              GateThresholds, gate)

STREETS = {"0": "preflop", "1": "flop", "2": "turn", "3": "river"}


def acceptance(table: ConfidenceTable, th: GateThresholds) -> dict:
    n = defaultdict(Counter)
    w = defaultdict(Counter)
    reasons = defaultdict(Counter)
    first_action = defaultdict(Counter)
    for key, st in table.stats.items():
        street = STREETS.get(key.split("|", 1)[0], "?")
        d = gate(st, th, pathological=key in table.pathological)
        n[street][d.status] += 1
        w[street][d.status] += st.visits
        for r in d.reasons:
            reasons[street][r] += 1
        parts = key.split("|")
        if street == "preflop" and len(parts) > 3 and "nr0" in parts[3]:
            first_action["preflop first action"][d.status] += 1
    out = {}
    for street in ("preflop", "flop", "turn", "river"):
        tot, wt = sum(n[street].values()), sum(w[street].values())
        if not tot:
            continue
        out[street] = {
            "keys": tot,
            "share_keys": {s: round(n[street][s] / tot, 4) for s in (ACCEPT, LOW, REJECT)},
            "share_visit_weighted": {s: round(w[street][s] / wt, 4) for s in (ACCEPT, LOW, REJECT)},
            "reason_counts": dict(reasons[street].most_common()),
        }
    tot = sum(sum(c.values()) for c in n.values())
    wt = sum(sum(c.values()) for c in w.values())
    out["all"] = {"keys": tot,
                  "share_keys": {s: round(sum(n[k][s] for k in n) / tot, 4) for s in (ACCEPT, LOW, REJECT)},
                  "share_visit_weighted": {s: round(sum(w[k][s] for k in w) / wt, 4)
                                           for s in (ACCEPT, LOW, REJECT)}}
    out["preflop_first_action_keys"] = dict(first_action["preflop first action"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("table", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    table = ConfidenceTable.load(a.table)
    th = GateThresholds.calibrated()
    tpath = a.table.resolve()
    shown = tpath.relative_to(ROOT) if tpath.is_relative_to(ROOT) else tpath.name
    res = {"format": "pokeralpha.gate_acceptance/v1", "table": str(shown),
           "config_signature": table.config_signature, "table_meta": table.meta,
           "thresholds": th.__dict__, "by_street": acceptance(table, th)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["by_street"].items():
        if isinstance(v, dict) and "share_visit_weighted" in v:
            print(k, v["keys"], v["share_visit_weighted"])
    print("preflop first action:", res["by_street"]["preflop_first_action_keys"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
