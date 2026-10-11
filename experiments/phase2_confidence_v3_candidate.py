"""Phase II Programs 47-49: is there a confidence-v3 candidate?

Interpretable logistic regression on the existing signals (log visits, seed
disagreement, recent movement, their missingness) predicting "regret >= 2%
pot". Evaluated strictly held-out:

  LOGO     train on 8 games, score the held-out game; accept the most
           confident rows at the SAME coverage the v2 rule achieves on that
           game, and compare accepted regret / false confidence to v2.
  SHIFT    train on one family (preflop or river), test on the other.

v3 would only be proposed with a clear held-out win on multiple games and no
leakage; otherwise v2 stays (Program 47 rule).

    python experiments/phase2_confidence_v3_candidate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from phase2_confidence_ablation import RELEASE, accept_mask, load_rows  # noqa: E402

OUT = ROOT / "results" / "phase2" / "confidence_v3_candidate.json"


def features(rows):
    X = []
    for r in rows:
        sd = r.get("seed_disagreement")
        mv = r.get("mv_recent")
        X.append([1.0, np.log1p(r["visits"]), 0.0 if sd is None else sd, float(sd is None),
                  0.0 if mv is None else mv, float(mv is None)])
    return np.array(X)


def labels(rows):
    return np.array([float(r["ev_regret"] / r["pot"] >= 0.02) for r in rows])


def fit_logistic(X, y, l2=1e-2, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def evaluate(rows, score, coverage):
    """Accept the top `coverage` fraction by confidence (= -risk score)."""
    reg = np.array([r["ev_regret"] / r["pot"] for r in rows])
    wrong = np.array([r["wrong_best"] for r in rows])
    k = max(1, int(round(coverage * len(rows))))
    idx = np.argsort(score)[:k]          # lowest predicted risk first
    return {"coverage": k / len(rows), "mean_regret_pot": float(reg[idx].mean()), "false_confidence_rate": float(wrong[idx].mean())}


def main():
    _, rows = load_rows()
    games = sorted({r["game"] for r in rows})
    doc = {"format": "pokeralpha.phase2.confidence_v3_candidate/v1",
           "model": "L2 logistic regression on [1, log1p(visits), seed_disagreement, sd_missing, mv_recent, mv_missing]",
           "target": "ev_regret >= 2% of pot", "logo": {}, "shift": {}}
    wins = 0
    for g in games:
        tr = [r for r in rows if r["game"] != g]
        te = [r for r in rows if r["game"] == g]
        w = fit_logistic(features(tr), labels(tr))
        risk = features(te) @ w
        v2 = accept_mask(te, "v2_full", RELEASE)
        cov = float(v2.mean())
        if cov == 0:
            doc["logo"][g] = {"v2_coverage": 0.0, "note": "v2 accepts nothing in this game"}
            continue
        reg = np.array([r["ev_regret"] / r["pot"] for r in te])
        wrong = np.array([r["wrong_best"] for r in te])
        v2m = {"coverage": cov, "mean_regret_pot": float(reg[v2].mean()), "false_confidence_rate": float(wrong[v2].mean())}
        lr = evaluate(te, risk, cov)
        better = lr["mean_regret_pot"] < v2m["mean_regret_pot"] * 0.9 and lr["false_confidence_rate"] <= v2m["false_confidence_rate"]
        wins += better
        doc["logo"][g] = {"v2": v2m, "logistic_at_v2_coverage": lr, "logistic_clearly_better": bool(better), "weights": w.tolist()}
    doc["logo_summary"] = {"games_with_v2_acceptances": sum(1 for v in doc["logo"].values() if "v2" in v),
                           "games_logistic_clearly_better": wins}
    for train_f, test_f in (("preflop", "river"), ("river", "preflop")):
        tr = [r for r in rows if r["family"] == train_f]
        te = [r for r in rows if r["family"] == test_f]
        w = fit_logistic(features(tr), labels(tr))
        risk = features(te) @ w
        v2 = accept_mask(te, "v2_full", RELEASE)
        cov = float(v2.mean())
        reg = np.array([r["ev_regret"] / r["pot"] for r in te])
        wrong = np.array([r["wrong_best"] for r in te])
        # in-family fit for reference (optimistic)
        w_in = fit_logistic(features(te), labels(te))
        doc["shift"][f"{train_f}->{test_f}"] = {
            "v2": {"coverage": cov, "mean_regret_pot": float(reg[v2].mean()) if v2.any() else None,
                   "false_confidence_rate": float(wrong[v2].mean()) if v2.any() else None},
            "logistic_shifted": evaluate(te, risk, cov) if cov > 0 else None,
            "logistic_in_family_optimistic": evaluate(te, features(te) @ w_in, cov) if cov > 0 else None,
            "weights_shifted": w.tolist(), "weights_in_family": w_in.tolist()}
    doc["decision"] = ("PROPOSE v3 CANDIDATE" if wins >= 5 else "KEEP v2: no clear held-out win across games")
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    for g, v in doc["logo"].items():
        if "v2" in v:
            print(f"{g:24s} v2 cov {v['v2']['coverage']:.3f} reg {v['v2']['mean_regret_pot']:.4f} fc {v['v2']['false_confidence_rate']:.3f} | "
                  f"LR reg {v['logistic_at_v2_coverage']['mean_regret_pot']:.4f} fc {v['logistic_at_v2_coverage']['false_confidence_rate']:.3f} {'WIN' if v['logistic_clearly_better'] else ''}")
        else:
            print(g, v)
    for k, v in doc["shift"].items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if not kk.startswith("weights")}, default=float))
    print(doc["decision"])


if __name__ == "__main__":
    main()
