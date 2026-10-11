"""Phase II Program 4: confidence-system ablation on the exact-game
calibration dataset.

Data: results/validation/confidence_calibration_dataset.json (8,880 infoset
rows from 9 exact games x milestones x seeds, each with the production
signals -- visits, recent movement, seed disagreement -- and the TRUE
EV regret / strategy L1 / wrong-best-action of the solver's policy).

Question: which signals actually contribute to accepted-risk performance?

Rules evaluated (ACCEPT iff ...):
  always_accept            everything
  visits_only              visits >= v
  seed_only                seed_disagreement < s
  movement_only            mv_recent < m
  visits_seed              visits >= v and seed_disagreement < s
  v2_full                  visits >= v and sd < s and mv_recent < m   (release rule: 20 / 0.2 / 0.1)
  v2_minus_<signal>        v2_full with one signal removed
  v1_gate                  historical v1 status == SOLVER_ACCEPT
  collision                NOT APPLICABLE: NaN by construction (perfect-recall keys in the exact games)

Two protocols:
  fixed      the promoted thresholds (20 / 0.2 / 0.1), evaluated per held-out game
  nested     for each held-out game, thresholds are chosen on the other games
             (grid search, objective: lowest accepted mean regret subject to
             coverage >= 15%), then applied to the held-out game -- so the
             held-out number never saw its own thresholds.

Metrics (defined in the output): coverage, accepted mean regret (pot units
of each game family: 2 bb preflop, 10 chips river), accepted tail regret
(95th percentile), false-confidence rate (accepted and wrong best action),
AUC of each continuous signal for regret >= 2% pot, bootstrap CIs over
rows and a leave-one-game-out jackknife over games.

    python experiments/phase2_confidence_ablation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confidence_signal_quality import auc  # noqa: E402

DATA = ROOT / "results" / "validation" / "confidence_calibration_dataset.json"
OUT = ROOT / "results" / "phase2" / "confidence_ablation.json"
RELEASE = {"v": 20.0, "s": 0.2, "m": 0.1}
GRID = {"v": [0.0, 20.0, 50.0, 100.0], "s": [0.1, 0.2, 0.3, 0.4, 0.5, 1e9], "m": [0.05, 0.1, 0.2, 0.3, 1e9]}
SIGNALS = ("visits", "seed_disagreement", "mv_recent")


def load_rows():
    d = json.loads(DATA.read_text())
    rows = [r for r in d["rows"] if r.get("l1_error") is not None and r.get("ev_regret") is not None]
    for r in rows:
        r["pot"] = 10.0 if r["family"] == "river" else 2.0
    return d, rows


def accept_mask(rows, rule, th):
    v, s, m = th["v"], th["s"], th["m"]
    out = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        vis = r["visits"]
        sd = r.get("seed_disagreement")
        mv = r.get("mv_recent")
        c_v = vis >= v
        c_s = sd is not None and sd < s
        c_m = mv is None or mv < m
        if rule == "always_accept":
            ok = True
        elif rule == "visits_only":
            ok = c_v
        elif rule == "seed_only":
            ok = c_s
        elif rule == "movement_only":
            ok = c_m
        elif rule == "visits_seed":
            ok = c_v and c_s
        elif rule == "v2_full":
            ok = c_v and c_s and c_m
        elif rule == "v2_minus_visits":
            ok = c_s and c_m
        elif rule == "v2_minus_seed":
            ok = c_v and c_m
        elif rule == "v2_minus_movement":
            ok = c_v and c_s
        elif rule == "v1_gate":
            ok = r.get("v1_status") == "SOLVER_ACCEPT"
        else:
            raise ValueError(rule)
        out[i] = ok
    return out


def metrics(rows, mask):
    reg = np.array([r["ev_regret"] / r["pot"] for r in rows])
    wrong = np.array([r["wrong_best"] for r in rows])
    n = int(mask.sum())
    if n == 0:
        return {"coverage": 0.0, "accepted": 0, "mean_regret_pot": None, "tail95_regret_pot": None,
                "false_confidence_rate": None}
    return {"coverage": float(mask.mean()), "accepted": n,
            "mean_regret_pot": float(reg[mask].mean()),
            "tail95_regret_pot": float(np.percentile(reg[mask], 95)),
            "false_confidence_rate": float(wrong[mask].mean())}


def evaluate_rule(rows, rule, th):
    return metrics(rows, accept_mask(rows, rule, th))


def bootstrap(rows, rule, th, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    mask = accept_mask(rows, rule, th)
    reg = np.array([r["ev_regret"] / r["pot"] for r in rows])
    wrong = np.array([r["wrong_best"] for r in rows])
    cov, mr, fc = [], [], []
    N = len(rows)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        mk = mask[idx]
        cov.append(mk.mean())
        if mk.sum():
            mr.append(reg[idx][mk].mean())
            fc.append(wrong[idx][mk].mean())
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x else None  # noqa: E731
    return {"coverage_ci95": pct(cov), "mean_regret_ci95": pct(mr), "false_confidence_ci95": pct(fc)}


def nested_select(train_rows, rule, min_coverage=0.15):
    """Grid-search thresholds for ``rule`` on the training games."""
    best = None
    for v in GRID["v"]:
        for s in GRID["s"]:
            for m in GRID["m"]:
                th = {"v": v, "s": s, "m": m}
                mt = evaluate_rule(train_rows, rule, th)
                if mt["mean_regret_pot"] is None or mt["coverage"] < min_coverage:
                    continue
                key = (mt["mean_regret_pot"], -mt["coverage"])
                if best is None or key < best[0]:
                    best = (key, th)
    return best[1] if best else dict(RELEASE)


def main():
    t0 = time.time()
    data, rows = load_rows()
    games = sorted({r["game"] for r in rows})
    rules = ["always_accept", "visits_only", "seed_only", "movement_only", "visits_seed", "v2_full",
             "v2_minus_visits", "v2_minus_seed", "v2_minus_movement", "v1_gate"]
    doc = {"format": "pokeralpha.phase2.confidence_ablation/v1", "rows": len(rows), "games": games,
           "release_thresholds": RELEASE, "grid": GRID,
           "metric_definitions": {
               "coverage": "fraction of calibration rows the rule accepts",
               "mean_regret_pot": "mean true EV regret of accepted rows, in pots (preflop pot 2 bb, river pot 10 chips)",
               "tail95_regret_pot": "95th percentile of accepted true EV regret, in pots",
               "false_confidence_rate": "fraction of accepted rows whose solver best action is not the true best action",
               "auc": "ROC-AUC of the signal for classifying rows with regret >= 2% pot (larger signal = less confident)"},
           "fixed": {}, "nested": {}, "per_game_fixed": {}, "ablation_deltas": {}, "signal_auc": {}, "jackknife": {}}
    # fixed thresholds, all rows + bootstrap
    for rule in rules:
        mt = evaluate_rule(rows, rule, RELEASE)
        mt.update(bootstrap(rows, rule, RELEASE))
        doc["fixed"][rule] = mt
        doc["per_game_fixed"][rule] = {g: evaluate_rule([r for r in rows if r["game"] == g], rule, RELEASE) for g in games}
    # ablation deltas vs v2_full (fixed)
    full = doc["fixed"]["v2_full"]
    for rule in ("v2_minus_visits", "v2_minus_seed", "v2_minus_movement", "visits_seed"):
        m = doc["fixed"][rule]
        doc["ablation_deltas"][rule] = {k: (None if m[k] is None or full[k] is None else m[k] - full[k])
                                        for k in ("coverage", "mean_regret_pot", "tail95_regret_pot", "false_confidence_rate")}
    # nested LOGO: thresholds chosen on the other games, applied to the held-out game
    for rule in rules:
        if rule in ("always_accept", "v1_gate"):
            continue
        acc_reg, acc_wrong, n_acc, n_tot, chosen = [], [], 0, 0, {}
        for g in games:
            train = [r for r in rows if r["game"] != g]
            test = [r for r in rows if r["game"] == g]
            th = nested_select(train, rule)
            chosen[g] = th
            mk = accept_mask(test, rule, th)
            n_tot += len(test)
            n_acc += int(mk.sum())
            acc_reg += [r["ev_regret"] / r["pot"] for r, k in zip(test, mk) if k]
            acc_wrong += [r["wrong_best"] for r, k in zip(test, mk) if k]
        doc["nested"][rule] = {"coverage": n_acc / n_tot, "accepted": n_acc,
                               "mean_regret_pot": float(np.mean(acc_reg)) if acc_reg else None,
                               "tail95_regret_pot": float(np.percentile(acc_reg, 95)) if acc_reg else None,
                               "false_confidence_rate": float(np.mean(acc_wrong)) if acc_wrong else None,
                               "thresholds_per_heldout_game": chosen}
    # AUCs of continuous signals
    reg = np.array([r["ev_regret"] for r in rows])
    pot = np.array([r["pot"] for r in rows])
    label = (reg >= 0.02 * pot).astype(int)
    for sig, direction in (("visits", -1), ("seed_disagreement", 1), ("mv_recent", 1), ("mv_hist", 1), ("action_margin", -1)):
        xs = np.array([r.get(sig) if r.get(sig) is not None else np.nan for r in rows], dtype=float)
        ok = ~np.isnan(xs)
        doc["signal_auc"][sig] = {"auc_regret_2pct_pot": auc(direction * xs[ok], label[ok]), "n": int(ok.sum())}
    # jackknife over games for v2_full and visits_seed (fixed thresholds): leave one game out, recompute
    for rule in ("v2_full", "visits_seed", "seed_only", "v1_gate"):
        vals = []
        for g in games:
            sub = [r for r in rows if r["game"] != g]
            vals.append(evaluate_rule(sub, rule, RELEASE)["mean_regret_pot"])
        vals = np.array([v for v in vals if v is not None])
        n = len(vals)
        doc["jackknife"][rule] = {"mean_regret_pot_leave_one_game_out": vals.tolist(),
                                  "jackknife_se": float(np.sqrt((n - 1) / n * ((vals - vals.mean()) ** 2).sum()))}
    doc["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    doc["conclusions"] = []
    f = doc["fixed"]
    doc["conclusions"].append(
        f"fixed release thresholds: v2_full coverage {f['v2_full']['coverage']:.3f}, accepted regret "
        f"{f['v2_full']['mean_regret_pot']:.4f} pot, false-confidence {f['v2_full']['false_confidence_rate']:.3f}; "
        f"always_accept regret {f['always_accept']['mean_regret_pot']:.4f}, v1 gate regret {f['v1_gate']['mean_regret_pot']:.4f} "
        f"at coverage {f['v1_gate']['coverage']:.3f}")
    for rule, d in doc["ablation_deltas"].items():
        doc["conclusions"].append(f"{rule}: coverage {d['coverage']:+.3f}, accepted regret {d['mean_regret_pot']:+.4f} pot, "
                                  f"false-confidence {d['false_confidence_rate']:+.3f} vs v2_full")
    OUT.write_text(json.dumps(doc, indent=1))
    for rule in rules:
        m = doc["fixed"][rule]
        nm = doc["nested"].get(rule)
        print(f"{rule:20s} fixed: cov {m['coverage']:.3f} reg {m['mean_regret_pot']:.4f} tail {m['tail95_regret_pot']:.4f} "
              f"fc {m['false_confidence_rate']:.3f}" + (f" | nested: cov {nm['coverage']:.3f} reg {nm['mean_regret_pot']:.4f} fc {nm['false_confidence_rate']:.3f}" if nm else ""))
    print("auc:", {k: (None if v["auc_regret_2pct_pot"] is None else round(v["auc_regret_2pct_pot"], 3)) for k, v in doc["signal_auc"].items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
