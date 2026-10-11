"""Phase II Program 11: one full autonomous loop on real, confirmed weaknesses.

  python experiments/phase2_research_loop.py plan      # measure, flag, classify, write manifests
  python experiments/phase2_research_loop.py run       # after phase2_extend_training.py: run + compare + record

Experiments executed (from the rule-table diagnoses):
  W1 / extend_training_remeasure_states - solver-policy jump under a +10%
     faced-bet (or pot) perturbation, for the confirmed local-sensitivity
     failure states vs matched control states, at 2M vs 3M vs 4M iterations
     (seeds 0-2). Hypothesis (training coverage): failing states' jumps shrink
     more than controls'.
  W2 / extend_training_measure_disagreement - mean pairwise seed TV at 2M vs
     4M by visit bucket; pure sampling noise predicts ratio ~0.71.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.research import loop, runner  # noqa: E402

CAND = ROOT / "results" / "strategy" / "candidates"
EXT = CAND / "phase2_extension"
OUT = ROOT / "results" / "phase2" / "loop"


def artifact(seed, its):
    if its == 2000:
        return (ROOT / "results/strategy/holdem_v2_native_seed0_2m.npz") if seed == 0 else \
            CAND / "native_2m" / f"holdem_v2_native_seed{seed}_2000k.npz"
    return EXT / f"holdem_v2_native_seed{seed}_{its}k.npz"


def do_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    p = loop.plan()
    (OUT / "plan.json").write_text(json.dumps(p, indent=1, default=float))
    for wid, ds in p["diagnoses"].items():
        for d in ds:
            if d["experiment"]:
                m = loop.manifest(loop.Diagnosis(**d), {"seeds": [0, 1, 2], "from_iterations": 2_000_000,
                                                       "to_iterations": [3_000_000, 4_000_000],
                                                       "script": "experiments/phase2_extend_training.py + phase2_research_loop.py run"})
                (OUT / f"manifest_{wid}_{d['experiment']}.json").write_text(json.dumps(m, indent=1))
    for wid, ds in p["diagnoses"].items():
        print(wid, [(d["rule"], d["likely_source"], d["experiment"]) for d in ds])
    return p


def policies(path):
    z = np.load(path, allow_pickle=False)
    keys, off, probs, vis = z["keys"], z["action_offsets"], z["probs"].astype(np.float64), z["visits"]
    out = {}
    for i, k in enumerate(keys):
        p = probs[off[i]:off[i + 1]]
        out[str(k)] = (tuple(z["actions"][off[i]:off[i + 1]]), p / p.sum() if p.sum() > 0 else p, float(vis[i]))
    return out


def tv(a, b):
    if a is None or b is None:
        return None
    if a[0] != b[0]:
        # different legal sets after perturbation: compare on shared action labels
        da, db = dict(zip(a[0], a[1])), dict(zip(b[0], b[1]))
        return 0.5 * sum(abs(da.get(x, 0.0) - db.get(x, 0.0)) for x in set(da) | set(db))
    return 0.5 * float(np.abs(a[1] - b[1]).sum())


def state_pairs(recs, provider):
    """(key, perturbed key) for each record, perturbation as in the red team."""
    from poker_alpha.analysis.whatif import WhatIfError, apply_whatif
    from poker_alpha.pipeline import observe_manual
    out = []
    for r in recs:
        st = r["state"]
        try:
            hero = st["hero_seat"]
            bets = [s.get("bet", 0.0) for s in st["seats"]]
            facing = max(b for i, b in enumerate(bets) if i != hero) - bets[hero]
            pert = apply_whatif(st, "villain_bet", round(max(b for i, b in enumerate(bets) if i != hero) * 1.1, 2)) \
                if facing > 0 else apply_whatif(st, "pot", round(st["pot"] * 1.1, 2))
            a = provider._to_state(observe_manual(st).state)
            b = provider._to_state(observe_manual(pert).state)
            if isinstance(a, tuple) and isinstance(b, tuple):
                out.append((provider.game.infoset_key(a[0]), provider.game.infoset_key(b[0])))
        except (WhatIfError, ValueError, KeyError):
            continue
    return out


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def experiment_w1():
    from poker_alpha.decision.strategy import SolverStrategyProvider
    with gzip.open(ROOT / "results/phase2/redteam_discovery_records.json.gz", "rt") as fh:
        recs = json.load(fh)
    failing = [r for r in recs if r["fail"]["local_sensitivity"]]
    clusters = {r["cluster"] for r in failing}
    pool = [r for r in recs if not r["fail"]["local_sensitivity"] and r["cluster"] in clusters
            and r["local_sensitivity"] is not None]
    random.Random(0).shuffle(pool)
    control = pool[: 4 * len(failing)]
    prov = SolverStrategyProvider.from_artifact(artifact(0, 2000), use_gate=False)
    pf, pc = state_pairs(failing, prov), state_pairs(control, prov)
    res = {"failing_states": len(failing), "control_states": len(control), "failing_pairs": len(pf), "control_pairs": len(pc),
           "clusters": sorted(clusters), "by_artifact": {}}
    for seed in (0, 1, 2):
        for its in (2000, 3000, 4000):
            pth = artifact(seed, its)
            if not pth.exists():
                continue
            pol = policies(pth)
            jf = [tv(pol.get(a), pol.get(b)) for a, b in pf]
            jc = [tv(pol.get(a), pol.get(b)) for a, b in pc]
            jf = [x for x in jf if x is not None]
            jc = [x for x in jc if x is not None]
            vf = [pol[a][2] for a, _ in pf if a in pol]
            res["by_artifact"][f"seed{seed}_{its}k"] = {
                "failing_mean_jump": float(np.mean(jf)), "failing_ci95": boot_ci(jf), "n_failing": len(jf),
                "control_mean_jump": float(np.mean(jc)), "control_ci95": boot_ci(jc), "n_control": len(jc),
                "failing_median_visits": float(np.median(vf)) if vf else None}
    # paired change 2M -> 4M per seed, failing vs control
    deltas = {}
    for seed in (0, 1, 2):
        a, b = res["by_artifact"].get(f"seed{seed}_2000k"), res["by_artifact"].get(f"seed{seed}_4000k")
        if a and b:
            deltas[f"seed{seed}"] = {"failing_change": b["failing_mean_jump"] - a["failing_mean_jump"],
                                     "control_change": b["control_mean_jump"] - a["control_mean_jump"]}
    res["change_2m_to_4m"] = deltas
    fc = [d["failing_change"] for d in deltas.values()]
    cc = [d["control_change"] for d in deltas.values()]
    res["classification"] = ("POSITIVE" if fc and all(f < c - 0.02 for f, c in zip(fc, cc)) else
                             "NEGATIVE" if fc and all(f > c + 0.02 for f, c in zip(fc, cc)) else "NULL")
    res["classification_rule"] = "POSITIVE iff on every seed the failing set's mean jump fell >0.02 TV more than the control set's"
    return res


def experiment_w2():
    out = {}
    for its in (2000, 4000):
        pols = [policies(artifact(s, its)) for s in (0, 1, 2)]
        out[its] = pols
    vis = {k: v[2] for k, v in out[4000][0].items()}
    rows = {}
    for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 1e12)):
        r2, r4 = [], []
        for k, v in vis.items():
            if not (lo <= v < hi):
                continue
            if not all(k in p for p in out[2000] + out[4000]):
                continue
            r2.append(np.mean([tv(out[2000][i][k], out[2000][j][k]) for i, j in combinations(range(3), 2)]))
            r4.append(np.mean([tv(out[4000][i][k], out[4000][j][k]) for i, j in combinations(range(3), 2)]))
        if r2:
            rows[f"{int(lo)}-{'inf' if hi > 1e11 else int(hi)}"] = {"keys": len(r2), "seed_tv_2m": float(np.mean(r2)),
                                                                    "seed_tv_4m": float(np.mean(r4)),
                                                                    "ratio": float(np.mean(r4) / np.mean(r2))}
    allr = [v["ratio"] for v in rows.values()]
    return {"by_visit_bucket": rows, "noise_prediction_ratio": 1 / np.sqrt(2),
            "classification": "STRUCTURAL (training does not resolve it)" if allr and min(allr) > 0.85 else
            "MIXED" if allr and max(allr) > 0.8 else "NOISE-LIKE",
            "note": "visits from seed 0 at 4M; keys present in all six artifacts"}


def do_run():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    w1 = experiment_w1()
    print(json.dumps({k: v for k, v in w1.items() if k != "clusters"}, indent=1, default=float)[:3000])
    w2 = experiment_w2()
    print(json.dumps(w2, indent=1, default=float))
    doc = {"format": "pokeralpha.phase2.loop_results/v1", "W1_extend_training_remeasure_states": w1,
           "W2_extend_training_measure_disagreement": w2, "release_action": "none (human gate)",
           "seconds": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(doc, indent=1, default=float))
    runner.record("loop_W1_local_sensitivity_training", status="done", mode="confirmation",
                  outputs=["results/phase2/loop/results.json"], seeds=[0, 1, 2], classification=w1["classification"],
                  headline=w1["change_2m_to_4m"], command="python experiments/phase2_research_loop.py run")
    runner.record("loop_W2_seed_disagreement_4m", status="done", mode="confirmation",
                  outputs=["results/phase2/loop/results.json"], seeds=[0, 1, 2], classification="DESCRIPTIVE",
                  headline=w2["by_visit_bucket"], notes=w2["classification"] + "; researcher-directed follow-up of Program 18 (the loop rule table did not fire for W2)",
                  command="python experiments/phase2_research_loop.py run")


if __name__ == "__main__":
    {"plan": do_plan, "run": do_run}[sys.argv[1]]()
