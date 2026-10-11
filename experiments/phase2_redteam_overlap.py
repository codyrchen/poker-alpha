"""Phase II Program 10G: connect confirmed red-team failures to abstraction /
training signals at the KEY level (post-processing of the discovery records).

For every scored discovery state the abstract key is recomputed (also for
gate-rejected lookups) and joined with the release confidence table
(visits, recent movement, seed disagreement, collision). For each failure
objective: effect sizes (AUC of each key-level signal for separating failing
from non-failing states; rate ratios by street) with bootstrap CIs.

    python experiments/phase2_redteam_overlap.py
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_redteam import OBJECTIVES, solver_key  # noqa: E402

from poker_alpha.pipeline import load_solver  # noqa: E402
from poker_alpha.solver_config import RELEASE_CONFIDENCE, RELEASE_STRATEGY  # noqa: E402

OUT = ROOT / "results" / "phase2" / "redteam_abstraction_overlap.json"


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    if label.sum() in (0, len(label)):
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    r = np.arange(1, len(score) + 1, dtype=float)
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n1 = label.sum()
    n0 = len(label) - n1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main():
    solver = load_solver(RELEASE_STRATEGY)
    z = np.load(ROOT / RELEASE_CONFIDENCE, allow_pickle=False)
    conf = {str(k): row for k, row in zip(z["keys"], z["stats"])}
    with gzip.open(ROOT / "results" / "phase2" / "redteam_discovery_records.json.gz", "rt") as fh:
        recs = json.load(fh)
    rows = []
    for r in recs:
        k = solver_key(solver, r["state"])
        c = conf.get(k) if k else None
        rows.append({"street": r["street"], "fail": r["fail"], "key": k,
                     "k_visits": None if c is None else float(c[0]),
                     "k_movement": None if c is None or math.isnan(c[1]) else float(c[1]),
                     "k_seed": None if c is None or math.isnan(c[2]) else float(c[2]),
                     "k_collision": None if c is None or math.isnan(c[3]) else float(c[3])})
    have = [r for r in rows if r["key"] is not None]
    doc = {"format": "pokeralpha.phase2.redteam_abstraction_overlap/v1", "states": len(rows),
           "states_with_key": len(have), "states_with_confidence_stats": sum(r["k_visits"] is not None for r in rows),
           "collision_signal_coverage": {"release_keys_with_collision": int((~np.isnan(z["stats"][:, 3])).sum()),
                                         "release_keys": int(len(z["keys"])),
                                         "states_with_collision": sum(r["k_collision"] is not None for r in rows)},
           "objectives": {}}
    rng = np.random.default_rng(0)
    for o in OBJECTIVES:
        res = {}
        for sig, direction in (("k_visits", -1), ("k_seed", 1), ("k_movement", 1), ("k_collision", 1)):
            sub = [r for r in have if r[sig] is not None]
            if len(sub) < 50:
                res[sig] = {"n": len(sub), "auc": None}
                continue
            x = np.array([direction * r[sig] for r in sub])
            y = np.array([r["fail"][o] for r in sub], int)
            a = auc(x, y)
            boots = []
            for _ in range(200):
                idx = rng.integers(0, len(sub), len(sub))
                b = auc(x[idx], y[idx])
                if b is not None:
                    boots.append(b)
            res[sig] = {"n": len(sub), "failing": int(y.sum()), "auc": a,
                        "auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                        "direction": "low visits = suspicious" if direction < 0 else "high value = suspicious"}
        by_st = defaultdict(lambda: [0, 0])
        for r in rows:
            by_st[r["street"]][0] += r["fail"][o]
            by_st[r["street"]][1] += 1
        res["rate_by_street"] = {s: {"rate": a / b, "n": b} for s, (a, b) in by_st.items()}
        doc["objectives"][o] = res
        print(o, {s: (None if v.get("auc") is None else round(v["auc"], 3)) for s, v in res.items() if s.startswith("k_")},
              {s: round(v["rate"], 3) for s, v in res["rate_by_street"].items()})
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({k: v for k, v in doc.items() if k != "objectives"}, indent=1))


if __name__ == "__main__":
    main()
