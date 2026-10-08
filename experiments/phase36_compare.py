"""Phase 36: compare the v1 and v2 solver configs at matched iterations.

Reads committed analyses (holdem_training_v1.json at 100k, holdem_training_v2.json,
preflop_audit_v1.json, preflop_audit_v2.json) and two confidence tables built
the same way (final 100k vs earlier 10k, seeds 0-2). Writes
results/validation/solver_quality_v1.json section ``v1_vs_v2``.

Cross-play between v1 and v2 is not meaningful (different games: different
legal menus); each config's cross-play is within its own game.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision.solver_gate import ConfidenceTable, GateThresholds, gate  # noqa: E402

V = ROOT / "results" / "validation"


def load(n):
    return json.loads((V / n).read_text())


def preflop_stats(audit, it):
    out = {}
    sits = sorted({r["situation"] for r in audit["records"].values()})
    for sit in sits:
        recs = [r for r in audit["records"].values() if r["situation"] == sit]
        dis, jam, vis = [], [], []
        for r in recs:
            pols = [r["checkpoints"].get(f"s{s}@{it}") for s in (0, 1, 2)]
            pols = [p for p in pols if p]
            if len(pols) < 2:
                continue
            dis.append(np.mean([np.abs(np.array(a["avg"]) - np.array(b["avg"])).sum()
                                for i, a in enumerate(pols) for b in pols[i + 1:]]))
            mean = np.mean([p["avg"] for p in pols], axis=0)
            jam.append(mean[r["legal"].index("a")] if "a" in r["legal"] else 0.0)
            vis.append(np.mean([p["visits"] for p in pols]))
        out[sit] = {"mean_seed_disagreement": round(float(np.mean(dis)), 4) if dis else None,
                    "mean_jam_freq": round(float(np.mean(jam)), 4) if jam else None,
                    "median_visits": round(float(np.median(vis)), 1) if vis else None}
    return out


def hand(audit, sit, cls, it):
    r = audit["records"][f"{sit}|{cls}"]
    pols = [r["checkpoints"].get(f"s{s}@{it}") for s in (0, 1, 2)]
    pols = [p for p in pols if p]
    mean = np.mean([p["avg"] for p in pols], axis=0)
    return {"legal": r["legal"], "mean_avg": [round(float(x), 4) for x in mean],
            "per_seed": [p["avg"] for p in pols]}


def gate_stats(path):
    t = ConfidenceTable.load(path)
    th = GateThresholds.calibrated()
    by, w = Counter(), Counter()
    for k, st in t.stats.items():
        d = gate(st, th, k in t.pathological)
        by[d.status] += 1
        w[d.status] += st.visits
    tot, wt = sum(by.values()), sum(w.values())
    arr = np.array([[s.visits, s.movement, s.seed_disagreement] for s in t.stats.values()])
    return {"keys": tot, "share_keys": {k: round(v / tot, 4) for k, v in by.items()},
            "share_visit_weighted": {k: round(v / wt, 4) for k, v in w.items()},
            "median_seed_disagreement": round(float(np.nanmedian(arr[:, 2])), 4),
            "median_movement": round(float(np.nanmedian(arr[:, 1])), 4)}


def matrix_disagreement(train, it):
    pairs = train["seed_disagreement"][str(it)]["pairs"]
    return {k: {"matrix_both_ge20_mean_l1": v["matrix_both_ge20"]["mean_l1"],
                "top2000_overlap": v["top2000_overlap"],
                "top2000_common_mean_l1": v["top2000_common_mean_l1"]} for k, v in pairs.items()}


def main():
    t1, t2 = load("holdem_training_v1.json"), load("holdem_training_v2.json")
    a1, a2 = load("preflop_audit_v1.json"), load("preflop_audit_v2.json")
    it = 100000
    row = lambda t, s: [r for r in t["convergence_proxies"][s] if r["iterations"] == it][0]  # noqa: E731
    out = {"iterations": it, "note": "v1 = HoldemSolverConfig:v1 (pot-fraction sizes, 0..7 river rung); "
                                     "v2 = legal sizing + 20 exact river percentile buckets"}
    for name, t in (("v1", t1), ("v2", t2)):
        r = row(t, "0")
        out[name] = {
            "config_signature": t["config_signature"],
            "infosets_seed0": r["infosets"],
            "new_infosets_per_iteration": r["new_infosets_per_iteration"],
            "fraction_trained_ge5": r["visits"]["fraction_trained_ge5"],
            "fraction_trained_ge20": r["visits"]["fraction_trained_ge20"],
            "seed_disagreement": matrix_disagreement(t, it),
            "sanity_checks_passed": {s: sum(c["result"] == "PASS" for c in v)
                                     for s, v in t["sanity_checks"].items()},
            "crossplay": [{k: c[k] for k in ("match", "a", "b", "bb_per_100", "ci95_bb_per_100")}
                          for c in t["crossplay"]["matches"]],
        }
    out["v1"]["preflop"] = preflop_stats(a1, it)
    out["v2"]["preflop"] = preflop_stats(a2, it)
    for name, a in (("v1", a1), ("v2", a2)):
        out[name]["BTN_first_AA"] = hand(a, "BTN_first", "AA", it)
        out[name]["BTN_first_72o"] = hand(a, "BTN_first", "72o", it)
        out[name]["BTN_first_KK"] = hand(a, "BTN_first", "KK", it)
    for name, f in (("v1", "holdem_v1_100k_confidence.npz"), ("v2", "holdem_v2_seed0_confidence.npz")):
        p = ROOT / "results" / "strategy" / f
        if not p.exists():
            p = Path("/home/user/pa_ckpt") / f
        if p.exists():
            out[name]["gate_at_100k"] = gate_stats(p)
    doc = json.loads((V / "solver_quality_v1.json").read_text())
    doc["v1_vs_v2"] = out
    (V / "solver_quality_v1.json").write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({k: out[k] if k not in ("v1", "v2") else {kk: vv for kk, vv in out[k].items()
                                                                 if kk not in ("crossplay", "preflop")}
                      for k in out}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
