"""Phase II self-generated hypotheses (top-ranked, implemented).

H1 seed ensembling: is the per-infoset average of k independently seeded
   ES-MCCFR average strategies less exploitable than (a) a single seed at the
   same per-seed iterations and (b) a single seed run k times as long (equal
   total compute)? Exact games: Leduc, reduced preflop Hold'em, river subgame.
   Ensemble weighting: by each seed's average-strategy mass (strategy_sum
   totals), which is the realization weight MCCFR itself accumulates.

H2 imperfect recall under the RELEASE algorithm: Program 5 found CFR+ gets
   worse with training on imperfect-recall Leduc abstractions. Does
   ES-MCCFR with uniform averaging (the release algorithm) degrade the same
   way, and does Discounted CFR avoid it?

    python experiments/phase2_hypotheses.py --parts H1,H2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_abstraction_science import LEDUC_ABS, KeyedGame  # noqa: E402
from phase2_cfr_family_benchmark import make_game  # noqa: E402

from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.research.cfr_variants import DiscountedCFRSolver  # noqa: E402
from poker_alpha.solvers import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

OUT = ROOT / "results" / "phase2" / "hypotheses.json"


def ensemble(solvers):
    keys = set().union(*[set(s.infosets) for s in solvers])
    out = {}
    for k in keys:
        acc, tot, acts = None, 0.0, None
        for s in solvers:
            n = s.infosets.get(k)
            if n is None:
                continue
            acts = n.actions
            v = n.strategy_sum
            acc = v.copy() if acc is None else acc + v
            tot += v.sum()
        out[k] = {a: (float(acc[i] / tot) if tot > 0 else 1.0 / len(acts)) for i, a in enumerate(acts)}
    return out


def h1_job(args):
    game_name, base_iters, k, rep = args
    g = make_game(game_name)
    seeds = [1000 * rep + i for i in range(k)]
    sol = []
    for s in seeds:
        m = MCCFRSolver(g, seed=s)
        m.train(base_iters)
        sol.append(m)
    single = [exploitability(g, m.average_strategy()) for m in sol]
    ens = exploitability(g, ensemble(sol))
    long_ = MCCFRSolver(g, seed=10_000 + rep)
    long_.train(base_iters * k)
    return {"game": game_name, "base_iters": base_iters, "k": k, "rep": rep, "single_mean": float(np.mean(single)),
            "single_best": float(np.min(single)), "ensemble": ens, "single_k_times_longer": exploitability(g, long_.average_strategy())}


def h2_job(args):
    abs_name, algo, seed, checkpoints = args
    real = LeducPoker()
    fn, _ = LEDUC_ABS[abs_name]
    g = KeyedGame(real, fn, abs_name)
    s = MCCFRSolver(g, seed=seed) if algo == "es_mccfr" else (
        DiscountedCFRSolver.preset(g, "dcfr") if algo == "dcfr" else CFRPlusSolver(g))
    curve, done = [], 0
    for c in checkpoints:
        s.train(c - done)
        done = c
        ab = s.average_strategy()
        lifted = {}
        stack = [real.root()]
        while stack:
            st = stack.pop()
            if real.is_terminal(st):
                continue
            if real.is_chance(st):
                stack.extend(x for _, x in real.chance_outcomes(st))
                continue
            rk = real.infoset_key(st)
            if rk not in lifted:
                acts = real.legal_actions(st)
                lifted[rk] = ab.get(fn(real, st), {a: 1 / len(acts) for a in acts})
            stack.extend(real.next_state(st, a) for a in real.legal_actions(st))
        curve.append({"iterations": c, "exploitability": exploitability(real, lifted)})
    return {"abstraction": abs_name, "algorithm": algo, "seed": seed, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="H1,H2")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"format": "pokeralpha.phase2.hypotheses/v1"}
    with Pool(a.workers) as pool:
        if "H1" in a.parts:
            jobs = [(g, it, k, rep) for g, it in (("leduc", 20000), ("reduced_preflop", 20000), ("river", 20000))
                    for k in (3,) for rep in range(8)]
            res = pool.map(h1_job, jobs, chunksize=1)
            summ = {}
            for g in ("leduc", "reduced_preflop", "river"):
                rs = [r for r in res if r["game"] == g]
                e = np.array([r["ensemble"] for r in rs])
                s = np.array([r["single_mean"] for r in rs])
                lng = np.array([r["single_k_times_longer"] for r in rs])
                summ[g] = {"reps": len(rs), "single_seed_mean": float(s.mean()), "ensemble_of_3_mean": float(e.mean()),
                           "single_3x_longer_mean": float(lng.mean()),
                           "ensemble_vs_single_ratio": float(np.mean(e / s)),
                           "ensemble_vs_3x_longer_ratio": float(np.mean(e / lng)),
                           "ensemble_beats_3x_longer": int((e < lng).sum())}
                print("H1", g, summ[g], flush=True)
            doc["H1_seed_ensembling"] = {"question": "does averaging 3 independent seeds beat 1 seed, and 1 seed at 3x the iterations?",
                                         "summary": summ, "runs": res}
        if "H2" in a.parts:
            cps = [100, 300, 1000, 3000, 10000, 30000, 100000]
            jobs = [(ab, "es_mccfr", s, cps) for ab in ("perfect", "forget_round1_history", "round1_raise_count_only") for s in range(4)]
            jobs += [(ab, "dcfr", 0, [10, 30, 100, 300, 1000, 3000]) for ab in ("perfect", "forget_round1_history", "round1_raise_count_only")]
            res = pool.map(h2_job, jobs, chunksize=1)
            summ = {}
            for r in res:
                key = f"{r['abstraction']}|{r['algorithm']}"
                summ.setdefault(key, []).append([c["exploitability"] for c in r["curve"]])
            summ = {k: {"mean_curve": np.mean(v, axis=0).round(4).tolist(), "runs": len(v)} for k, v in summ.items()}
            for k, v in summ.items():
                print("H2", k, v["mean_curve"], flush=True)
            doc["H2_imperfect_recall_release_algorithm"] = {"es_mccfr_checkpoints": cps, "dcfr_checkpoints": [10, 30, 100, 300, 1000, 3000],
                                                            "summary": summ, "runs": res}
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1, default=float))


if __name__ == "__main__":
    main()
