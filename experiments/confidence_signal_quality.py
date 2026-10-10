"""Phase 7 + 21/22 (final-trust): signal quality and risk-coverage on the
true-error calibration dataset.

Reads results/validation/confidence_calibration_dataset.json and measures,
with leave-one-game-out (LOGO) evaluation:

* Spearman correlation of every signal with strategy L1 error, EV regret
  and wrong-best-action;
* ROC-AUC for classifying high-error infosets (L1 >= 0.5; regret >= 2% pot);
* risk-coverage curves: sort by signal, accept the most-confident fraction,
  report mean accepted regret / wrong-action rate;
* v1 gate baseline: risk & coverage of its ACCEPT bucket;
* candidate v2 rules (seed disagreement + recent movement + visit floor) at
  several thresholds, evaluated on held-out games only.

Writes results/validation/confidence_signal_quality.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "results/validation/confidence_calibration_dataset.json"
OUT = ROOT / "results/validation/confidence_signal_quality.json"

SIGNALS = ["visits", "mv_hist", "mv_med", "mv_recent", "mv_recent_per10k",
           "seed_disagreement"]
# direction: +1 = larger signal means LESS confident
DIRECTION = {"visits": -1, "mv_hist": 1, "mv_med": 1, "mv_recent": 1,
             "mv_recent_per10k": 1, "seed_disagreement": 1}


def rows_arrays(rows, signal):
    xs, l1, reg, wrong, pot = [], [], [], [], []
    for r in rows:
        v = r.get(signal)
        if v is None or r.get("l1_error") is None:
            continue
        xs.append(v)
        l1.append(r["l1_error"])
        reg.append(r["ev_regret"])
        wrong.append(r["wrong_best"])
        pot.append(10.0 if r["family"] == "river" else 2.0)  # pot proxy
    return (np.array(xs), np.array(l1), np.array(reg), np.array(wrong),
            np.array(pot))


def auc(score, label):
    """ROC-AUC via rank statistic."""
    if label.sum() == 0 or label.sum() == len(label):
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def risk_coverage(score_conf_desc, regret, wrong, points=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    """score sorted so FIRST = most confident."""
    order = np.argsort(score_conf_desc)[::-1]
    out = {}
    for c in points:
        k = max(1, int(len(order) * c))
        idx = order[:k]
        out[str(c)] = {"mean_regret": float(regret[idx].mean()),
                       "wrong_rate": float(wrong[idx].mean())}
    return out


def main() -> None:
    data = json.loads(DATA.read_text())
    rows = data["rows"]
    games = sorted({r["game"] for r in rows})
    doc = {"format": "pokeralpha.confidence_signal_quality/v1",
           "rows": len(rows), "games": games,
           "pooled": {}, "logo": {}, "risk_coverage": {}, "v1_baseline": {},
           "v2_candidates": {}}

    # ---- pooled correlations / AUC ---------------------------------------
    for sig in SIGNALS + ["action_margin"]:
        xs, l1, reg, wrong, pot = rows_arrays(rows, sig)
        if len(xs) < 100:
            continue
        doc["pooled"][sig] = {
            "n": len(xs),
            "spearman_l1": float(spearmanr(xs, l1).statistic),
            "spearman_regret": float(spearmanr(xs, reg).statistic),
            "spearman_wrong": float(spearmanr(xs, wrong).statistic),
            "auc_highL1_0.5": auc(DIRECTION.get(sig, 1) * xs, (l1 >= 0.5).astype(int)),
            "auc_regret_2pct_pot": auc(DIRECTION.get(sig, 1) * xs,
                                       (reg >= 0.02 * pot).astype(int)),
        }

    # ---- LOGO correlations ------------------------------------------------
    for sig in SIGNALS:
        per = []
        for g in games:
            sub = [r for r in rows if r["game"] != g]
            xs, l1, reg, wrong, _ = rows_arrays(sub, sig)
            if len(xs) > 50:
                per.append(float(spearmanr(xs, reg).statistic))
        doc["logo"][sig] = {"spearman_regret_mean": float(np.mean(per)),
                            "spearman_regret_min": float(np.min(per)),
                            "spearman_regret_max": float(np.max(per))}

    # ---- risk-coverage per signal ------------------------------------------
    for sig in SIGNALS:
        xs, l1, reg, wrong, _ = rows_arrays(rows, sig)
        conf = -DIRECTION[sig] * xs       # larger = more confident
        doc["risk_coverage"][sig] = risk_coverage(conf, reg, wrong)
    base_reg = np.array([r["ev_regret"] for r in rows if r.get("l1_error") is not None])
    base_wrong = np.array([r["wrong_best"] for r in rows if r.get("l1_error") is not None])
    doc["risk_coverage"]["_all_rows"] = {"mean_regret": float(base_reg.mean()),
                                         "wrong_rate": float(base_wrong.mean())}

    # ---- v1 gate baseline ----------------------------------------------------
    for status in ("SOLVER_ACCEPT", "SOLVER_LOW_CONFIDENCE", "SOLVER_REJECT"):
        sub = [r for r in rows if r["v1_status"] == status]
        if sub:
            doc["v1_baseline"][status] = {
                "share": len(sub) / len(rows),
                "mean_regret": float(np.mean([r["ev_regret"] for r in sub])),
                "wrong_rate": float(np.mean([r["wrong_best"] for r in sub])),
                "mean_l1": float(np.mean([r["l1_error"] for r in sub
                                          if r["l1_error"] is not None]))}

    # ---- v2 candidate rules, LOGO-evaluated ---------------------------------
    # rule: ACCEPT iff sd < t_sd and mv_recent < t_mv and visits >= vmin
    grids = {"sd": [0.2, 0.3, 0.4, 0.5], "mv": [0.1, 0.2, 0.3, 1e9],
             "vmin": [0, 20]}
    results = {}
    for t_sd in grids["sd"]:
        for t_mv in grids["mv"]:
            for vmin in grids["vmin"]:
                accept_reg, accept_wrong, n_acc, n_tot = [], [], 0, 0
                for g in games:          # held-out game g
                    sub = [r for r in rows if r["game"] == g]
                    for r in sub:
                        n_tot += 1
                        sd = r.get("seed_disagreement")
                        mv = r.get("mv_recent")
                        ok = (sd is not None and sd < t_sd
                              and (mv is None or mv < t_mv)
                              and r["visits"] >= vmin)
                        if ok:
                            n_acc += 1
                            accept_reg.append(r["ev_regret"])
                            accept_wrong.append(r["wrong_best"])
                name = f"sd<{t_sd}|mv<{t_mv if t_mv < 1e8 else 'inf'}|v>={vmin}"
                results[name] = {
                    "coverage": n_acc / n_tot,
                    "mean_regret": float(np.mean(accept_reg)) if accept_reg else None,
                    "wrong_rate": float(np.mean(accept_wrong)) if accept_wrong else None}
    doc["v2_candidates"] = results

    OUT.write_text(json.dumps(doc, indent=1))
    print("pooled (spearman vs EV regret):")
    for sig, st in doc["pooled"].items():
        print(f"  {sig:>18}: rho_regret {st['spearman_regret']:+.3f} "
              f"rho_l1 {st['spearman_l1']:+.3f} auc_reg {st['auc_regret_2pct_pot']}")
    print("v1 baseline:", json.dumps(doc["v1_baseline"], indent=1))
    best = sorted(((v["mean_regret"], k) for k, v in results.items()
                   if v["mean_regret"] is not None and v["coverage"] >= 0.15))[:8]
    print("best v2 rules by accepted regret (coverage >= .15):")
    for reg, k in best:
        print(f"  {k}: regret {reg:.4f} coverage {results[k]['coverage']:.2f} "
              f"wrong {results[k]['wrong_rate']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
