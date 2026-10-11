"""Phase II Program 10: red team discovery -> confirmation at scale.

Discovery: N legal states from random engine self-play (state seed A), scored
by poker_alpha.analysis.redteam.score_state against the release solver; six
separate objectives, never combined. Interpretable clusters
(street x SPR bucket x faced-bet bucket x gate) with Wilson 95% intervals on
per-objective failure rates.

Confirmation, two independent routes:
  1. cluster level: an INDEPENDENT state sample (seed B) - are the clusters
     that ranked worst in discovery still worse than the population, with
     the effect size measured on the fresh sample only (no selection bias)?
  2. state level: the top-K states per stochastic objective are re-scored
     with a different decision seed and 4x the equity/rollout samples; a
     state is CONFIRMED if its score stays above the failure threshold.

Abstraction connection (10G): are failing states enriched in high collision
dispersion or low visits (confidence-table signals of the release key)?

    python experiments/phase2_redteam.py [--n 20000 --confirm-n 10000]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "phase2"
THRESH = {"local_sensitivity": 0.5, "model_sensitivity": 0.5, "seed_disagreement": 0.5,
          "translation_distance": 0.3}
OBJECTIVES = ["gate_rejected", "seed_disagreement", "low_visits", "translation_distance",
              "local_sensitivity", "model_sensitivity"]

_SOLVER = None
_CONF = None


def _init():
    global _SOLVER, _CONF
    from poker_alpha.pipeline import load_solver
    from poker_alpha.solver_config import RELEASE_CONFIDENCE, RELEASE_STRATEGY
    _SOLVER = load_solver(RELEASE_STRATEGY)
    z = np.load(ROOT / RELEASE_CONFIDENCE, allow_pickle=False)
    _CONF = {str(k): tuple(float(x) for x in row) for k, row in zip(z["keys"], z["stats"])}


def solver_key(solver, st):
    """Abstract infoset key of the hero decision (also for gate-rejected
    lookups, whose LookupMiss carries no key)."""
    from poker_alpha.pipeline import observe_manual
    provider = getattr(solver, "provider", solver)
    try:
        res = provider._to_state(observe_manual(st).state)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(res, tuple):
        return None
    return provider.game.infoset_key(res[0])


def spr_bucket(spr):
    if spr is None:
        return "na"
    return "<1" if spr < 1 else "1-3" if spr < 3 else "3-8" if spr < 8 else ">=8"


def bet_bucket(facing, pot):
    if facing <= 0:
        return "unopened"
    f = facing / max(pot - facing, 1e-9)
    return "<0.4" if f < 0.4 else "0.4-0.9" if f < 0.9 else "0.9-2" if f < 2 else ">=2"


def failures(r):
    return {"gate_rejected": bool(r["gate_rejected"]),
            "seed_disagreement": r["seed_disagreement"] is not None and r["seed_disagreement"] >= THRESH["seed_disagreement"],
            "low_visits": r["visits"] is not None and r["visits"] < 20,
            "translation_distance": r["translation_distance"] is not None and r["translation_distance"] >= THRESH["translation_distance"],
            "local_sensitivity": r["local_sensitivity"] is not None and r["local_sensitivity"] >= THRESH["local_sensitivity"],
            "model_sensitivity": r["model_sensitivity"] is not None and r["model_sensitivity"] >= THRESH["model_sensitivity"]}


def score_chunk(args):
    states, eq, ro, dseed = args
    from poker_alpha.analysis.redteam import score_state
    from poker_alpha.decision import DecisionConfig
    from poker_alpha.pipeline import observe_manual
    cfg = DecisionConfig(equity_simulations=eq, rollout_simulations=ro, seed=dseed)
    out = []
    for st in states:
        try:
            sc = score_state(st, _SOLVER, cfg)
        except Exception as e:  # noqa: BLE001 - record, never hide
            out.append({"error": f"{type(e).__name__}: {e}", "state": st})
            continue
        if sc is None:
            continue
        key = solver_key(_SOLVER, st)
        conf = _CONF.get(key) if key else None
        d = asdict(sc)
        d["key"] = key
        d["collision"] = None if conf is None or math.isnan(conf[3]) else conf[3]
        d["conf_visits"] = None if conf is None else conf[0]
        d["cluster"] = f"{sc.street}|spr{spr_bucket(sc.spr)}|bet{bet_bucket(sc.facing_bet_bb, sc.pot_bb)}"
        d["fail"] = failures(d)
        out.append(d)
    return out


def run(states, eq, ro, dseed, workers):
    chunks = [states[i:i + 50] for i in range(0, len(states), 50)]
    with Pool(workers, initializer=_init) as pool:
        parts = pool.map(score_chunk, [(c, eq, ro, dseed) for c in chunks], chunksize=1)
    return [r for p in parts for r in p]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def cluster_table(rows, min_n=30):
    by = defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    base = {o: float(np.mean([r["fail"][o] for r in rows])) for o in OBJECTIVES}
    table = {}
    for c, rs in by.items():
        if len(rs) < min_n:
            continue
        t = {"n": len(rs)}
        for o in OBJECTIVES:
            k = sum(r["fail"][o] for r in rs)
            t[o] = {"rate": k / len(rs), "ci95": wilson(k, len(rs)), "lift": (k / len(rs)) / base[o] if base[o] > 0 else None}
        table[c] = t
    return base, table


def overlap(rows):
    """Failing vs non-failing: collision and visits (abstraction connection)."""
    out = {}
    for o in OBJECTIVES:
        if o in ("low_visits",):
            continue
        f = [r for r in rows if r["fail"][o]]
        nf = [r for r in rows if not r["fail"][o]]

        def stat(rs, key, fn):
            v = [r[key] for r in rs if r[key] is not None]
            return (fn(v) if v else None), len(v)
        col_f, nf_f = stat(f, "collision", np.mean)
        col_n, nf_n = stat(nf, "collision", np.mean)
        hc_f = np.mean([r["collision"] >= 3 / 7 for r in f if r["collision"] is not None]) if nf_f else None
        hc_n = np.mean([r["collision"] >= 3 / 7 for r in nf if r["collision"] is not None]) if nf_n else None
        lv_f = np.mean([r["visits"] is not None and r["visits"] < 100 for r in f]) if f else None
        lv_n = np.mean([r["visits"] is not None and r["visits"] < 100 for r in nf]) if nf else None
        out[o] = {"n_fail": len(f), "n_ok": len(nf),
                  "mean_collision_fail": col_f, "mean_collision_ok": col_n,
                  "high_collision_share_fail": None if hc_f is None else float(hc_f),
                  "high_collision_share_ok": None if hc_n is None else float(hc_n),
                  "visits_lt_100_share_fail": None if lv_f is None else float(lv_f),
                  "visits_lt_100_share_ok": None if lv_n is None else float(lv_n),
                  "collision_available_fail": nf_f, "collision_available_ok": nf_n}
    return out


def strip(r):
    keep = ("street", "spr", "facing_bet_bb", "pot_bb", "gate", "gate_reasons", "method", "top_action", "mix",
            "gate_rejected", "seed_disagreement", "visits", "translation_distance", "local_sensitivity",
            "model_sensitivity", "key", "collision", "cluster", "fail", "state")
    return {k: r[k] for k in keep if k in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--confirm-n", type=int, default=10000)
    ap.add_argument("--seed-a", type=int, default=101)
    ap.add_argument("--seed-b", type=int, default=202)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    from poker_alpha.analysis.redteam import sample_states
    t0 = time.time()
    sa = sample_states(a.n, seed=a.seed_a)
    disc = run(sa, 100, 100, 0, a.workers)
    errors = [r for r in disc if "error" in r]
    disc = [r for r in disc if "error" not in r]
    t_disc = time.time() - t0
    base, table = cluster_table(disc)
    # rank clusters per objective by Wilson lower bound (discovery)
    ranked = {}
    m = max(1, len(table))
    for o in OBJECTIVES:
        cs = sorted(table.items(), key=lambda kv: -(kv[1][o]["ci95"][0] or 0))
        ranked[o] = [{"cluster": c, "n": t["n"], **t[o],
                      "bonferroni_significant": (t[o]["ci95"][0] or 0) > base[o] and
                      wilson(int(round(t[o]["rate"] * t["n"])), t["n"], z=3.0 + 0.1 * math.log(m))[0] > base[o]}
                     for c, t in cs[:5]]
    disc_doc = {"format": "pokeralpha.phase2.redteam_discovery/v1", "seed": a.seed_a, "n_sampled": a.n,
                "n_scored": len(disc), "errors": len(errors), "error_examples": errors[:5],
                "wall_seconds": round(t_disc, 1), "workers": a.workers, "thresholds": THRESH,
                "objectives": OBJECTIVES, "population_failure_rates": base, "clusters": table,
                "top_clusters_by_objective": ranked,
                "discipline": "discovery: cluster ranks are selected and therefore optimistic; see redteam_confirmation.json",
                "overlap_with_abstraction_signals": overlap(disc)}
    (OUT / "redteam_discovery.json").write_text(json.dumps(disc_doc, indent=1, default=float))
    with gzip.open(OUT / "redteam_discovery_records.json.gz", "wt") as fh:
        json.dump([strip(r) for r in disc], fh, default=float)
    print("discovery", len(disc), "scored in", round(t_disc), "s; base", {k: round(v, 3) for k, v in base.items()}, flush=True)
    # ---- confirmation 1: independent sample, cluster level ---------------
    t1 = time.time()
    sb = sample_states(a.confirm_n, seed=a.seed_b)
    conf = [r for r in run(sb, 100, 100, 1, a.workers) if "error" not in r]
    base_b, table_b = cluster_table(conf, min_n=10)
    cluster_conf = {}
    for o in OBJECTIVES:
        rows = []
        for item in ranked[o][:3]:
            c = item["cluster"]
            tb = table_b.get(c)
            if tb is None:
                rows.append({"cluster": c, "discovery_rate": item["rate"], "confirmation": None})
                continue
            rb = tb[o]
            rows.append({"cluster": c, "discovery_rate": item["rate"], "discovery_n": item["n"],
                         "confirmation_rate": rb["rate"], "confirmation_ci95": rb["ci95"], "confirmation_n": tb["n"],
                         "confirmation_population_rate": base_b[o],
                         "confirmed_above_population": (rb["ci95"][0] or 0) > base_b[o],
                         "effect_size_rate_difference": rb["rate"] - base_b[o],
                         "lift": rb["lift"]})
        cluster_conf[o] = rows
    # ---- confirmation 2: state level re-score (stochastic objectives) -------
    state_conf = {}
    for o in ("local_sensitivity", "model_sensitivity"):
        cand = sorted([r for r in disc if r[o] is not None], key=lambda r: -r[o])[: a.topk]
        rescored = run([r["state"] for r in cand], 400, 400, 99, a.workers)
        pairs = [(c[o], r.get(o)) for c, r in zip(cand, rescored) if "error" not in r]
        confirmed = [y is not None and y >= THRESH[o] for _, y in pairs]
        state_conf[o] = {"candidates": len(cand), "rescored": len(pairs),
                         "confirmed": int(sum(confirmed)), "confirmation_rate": float(np.mean(confirmed)) if pairs else None,
                         "discovery_mean": float(np.mean([x for x, _ in pairs])) if pairs else None,
                         "confirmation_mean": float(np.mean([y for _, y in pairs if y is not None])) if pairs else None,
                         "shrinkage": "regression to the mean expected: discovery scores are the max of many noisy draws",
                         "strongest_confirmed": max(((y, c) for (x, y), c in zip(pairs, cand) if y is not None),
                                                    key=lambda t: t[0], default=(None, None))[1] and
                         strip(max(((y, c) for (x, y), c in zip(pairs, cand) if y is not None), key=lambda t: t[0])[1])}
    conf_doc = {"format": "pokeralpha.phase2.redteam_confirmation/v1", "discovery_seed": a.seed_a,
                "confirmation_seed": a.seed_b, "n_confirmation_scored": len(conf),
                "population_failure_rates_confirmation": base_b,
                "cluster_confirmation": cluster_conf, "state_confirmation": state_conf,
                "overlap_with_abstraction_signals_confirmation": overlap(conf),
                "rescore_config": {"equity_simulations": 400, "rollout_simulations": 400, "decision_seed": 99},
                "wall_seconds": round(time.time() - t1, 1),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (OUT / "redteam_confirmation.json").write_text(json.dumps(conf_doc, indent=1, default=float))
    # clusters artifact: confirmation-sample table (unbiased) for the website / report
    (OUT / "redteam_clusters.json").write_text(json.dumps({
        "format": "pokeralpha.phase2.redteam_clusters/v1",
        "dimensions": ["street", "SPR bucket", "faced bet / pot-before-bet bucket"],
        "source": "independent confirmation sample (unbiased rates)", "population": base_b,
        "clusters": table_b}, indent=1, default=float))
    print(json.dumps({"cluster_confirmation": cluster_conf, "state_confirmation": {k: {kk: vv for kk, vv in v.items() if kk != "strongest_confirmed"} for k, v in state_conf.items()}}, indent=1, default=float)[:6000])
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
