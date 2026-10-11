"""Phase II Program 17: strategy compiler / interpretability.

Hold'em: per legal-action set (top groups by visits), shallow distribution
trees on parsed key features; held-out (key-hash split) visit-weighted TV,
top-action agreement, leaves, lookup latency; extracted rules.
Leduc: the same tree class on Leduc key features, distilling the exact CFR+
solution; the VALUE consequence is exact exploitability of the distilled
strategy in Leduc.

    python experiments/phase2_strategy_compiler.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.research.distill import HOLDEM_FEATURES, DistributionTree, holdem_features  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

REL = ROOT / "results" / "strategy" / "holdem_v2_native_seed0_2m.npz"
OUT = ROOT / "results" / "phase2" / "strategy_compiler.json"


def is_test(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0


def holdem(depths=(0, 2, 4, 6, 8), groups=8):
    z = np.load(REL, allow_pickle=False)
    keys, off, probs, visits = z["keys"], z["action_offsets"], z["probs"].astype(np.float64), z["visits"].astype(np.float64)
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[str(k).split("|")[-1]].append(i)
    order = sorted(by, key=lambda g: -visits[by[g]].sum())[:groups]
    covered = sum(visits[by[g]].sum() for g in order) / visits.sum()
    res = {"groups": [], "visit_share_covered": float(covered)}
    agg = defaultdict(lambda: {"w": 0.0, "tv": 0.0, "agree": 0.0, "leaves": 0})
    for g in order:
        idx = by[g]
        feats, Y, W, test = [], [], [], []
        for i in idx:
            f = holdem_features(str(keys[i]))
            if f is None:
                continue
            p = probs[off[i]:off[i + 1]]
            feats.append(f)
            Y.append(p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p)))
            W.append(visits[i] + 1.0)
            test.append(is_test(str(keys[i])))
        X, Y, W, test = np.array(feats), np.array(Y), np.array(W), np.array(test)
        gres = {"legal": g, "infosets": int(len(X)), "visit_share": float(W.sum() / visits.sum()), "by_depth": {}}
        for d in depths:
            t = DistributionTree(max_depth=d, min_leaf_weight=50.0, feature_names=HOLDEM_FEATURES)
            t.fit(X[~test], Y[~test], W[~test])
            t0 = time.perf_counter()
            pred = t.predict(X[test])
            lat = (time.perf_counter() - t0) / max(1, test.sum())
            tv = 0.5 * np.abs(pred - Y[test]).sum(1)
            agree = (pred.argmax(1) == Y[test].argmax(1)).astype(float)
            w = W[test]
            gres["by_depth"][str(d)] = {"leaves": t.leaves, "test_tv_visit_weighted": float((w * tv).sum() / w.sum()),
                                       "test_top_action_agreement_visit_weighted": float((w * agree).sum() / w.sum()),
                                       "test_tv_unweighted": float(tv.mean()), "lookup_seconds": lat}
            a = agg[d]
            a["w"] += w.sum()
            a["tv"] += (w * tv).sum()
            a["agree"] += (w * agree).sum()
            a["leaves"] += t.leaves
            if d == 4 and len(res["groups"]) < 3:
                gres["rules_depth4"] = t.rules(g.split("."), max_rules=8)
        res["groups"].append(gres)
        print(g, {d: (round(v["test_tv_visit_weighted"], 3), v["leaves"]) for d, v in gres["by_depth"].items()}, flush=True)
    res["overall"] = {str(d): {"test_tv_visit_weighted": a["tv"] / a["w"], "test_top_action_agreement": a["agree"] / a["w"],
                               "total_leaves": a["leaves"]} for d, a in agg.items()}
    return res


def leduc_features(key: str):
    rank, pub, hist = key.split("|", 2)
    rnd = 1 if "/" in hist else 0
    cur = hist.split("/")[-1]
    p = -1 if pub == "?" else int(pub)
    return [int(rank), p, float(p == int(rank)) if p >= 0 else -1.0, float(p >= 0 and int(rank) > p) if p >= 0 else -1.0,
            rnd, cur.count("r"), float(cur.endswith("r")), len(cur), hist.split("/")[0].count("r")]


LEDUC_FEATURES = ["rank", "public", "paired", "above_public", "round", "raises_this_round", "facing_raise",
                  "actions_this_round", "round1_raises"]


def leduc(depths=(0, 1, 2, 3, 4, 5, 6, 8)):
    g = LeducPoker()
    s = CFRPlusSolver(g)
    s.train(1500)
    eq = s.average_strategy()
    base = exploitability(g, eq)
    by = defaultdict(list)
    for k, p in eq.items():
        by[tuple(p)].append(k)
    out = {"reference_exploitability": base, "infosets": len(eq), "by_depth": {}}
    for d in depths:
        strat, leaves = {}, 0
        for acts, ks in by.items():
            X = np.array([leduc_features(k) for k in ks], float)
            Y = np.array([[eq[k][a] for a in acts] for k in ks])
            t = DistributionTree(max_depth=d, min_leaf_weight=1.0, feature_names=LEDUC_FEATURES).fit(X, Y, np.ones(len(ks)))
            leaves += t.leaves
            for k, x in zip(ks, X):
                v = t.predict_one(x)
                strat[k] = {a: float(p) for a, p in zip(acts, v)}
        e = exploitability(g, strat)
        tv = float(np.mean([0.5 * sum(abs(strat[k][a] - eq[k][a]) for a in eq[k]) for k in eq]))
        out["by_depth"][str(d)] = {"leaves": leaves, "exploitability": e, "mean_tv_to_cfr": tv}
        print("leduc depth", d, leaves, round(e, 4), round(tv, 3), flush=True)
    return out


def main():
    t0 = time.time()
    doc = {"format": "pokeralpha.phase2.strategy_compiler/v1", "label": "APPROXIMATION ONLY; never a replacement for the release",
           "holdem": holdem(), "leduc_value_check": leduc(),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    doc["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps(doc["holdem"]["overall"], indent=1))
    for gr in doc["holdem"]["groups"][:2]:
        print(gr["legal"]); print("\n".join(gr.get("rules_depth4", [])))


if __name__ == "__main__":
    main()
