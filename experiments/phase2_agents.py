"""Phase II Program 15: agent research on Kuhn with the generic agent interface.

15B round-robin payoff matrix (seat-swapped, several match seeds, CIs)
15C non-transitivity: significant 3-cycles and the support of the empirical
    meta-game's Nash equilibrium (LP)
15D behavioral fingerprints from simulated play
15E regime shift: opponent switches tight -> aggressive mid-match; stationary
    Bayes vs recency-weighted Bayes vs fixed equilibrium, per window
Also: posterior calibration of the Bayes agent against each true archetype.

    python experiments/phase2_agents.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.agents.adapters import PolicyTableAgent, RandomAgent, ScriptedKuhnAgent  # noqa: E402
from poker_alpha.agents.adaptive import BayesArchetypeAgent, SwitchingAgent  # noqa: E402
from poker_alpha.agents.tournament import play_episode, play_match  # noqa: E402
from poker_alpha.games.kuhn import KuhnPoker  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import best_response  # noqa: E402

OUT = ROOT / "results" / "phase2" / "agents.json"
GAME = KuhnPoker()


def kuhn_keys():
    keys = []
    for c in range(3):
        for h in ("", "p", "b", "pb"):
            keys.append((f"{c}{h}", ["p", "b"]))
    return keys


def table_of(agent):
    rng = random.Random(0)
    return {k: {a: (1.0 if agent.choose_action(k, acts, rng) == a else 0.0) for a in acts} for k, acts in kuhn_keys()}


def equilibrium():
    s = CFRPlusSolver(GAME)
    s.train(3000)
    return s.average_strategy()


def build_pool():
    eq = equilibrium()
    tight, aggr = ScriptedKuhnAgent("tight"), ScriptedKuhnAgent("aggressive")
    uniform = {k: {a: 0.5 for a in acts} for k, acts in kuhn_keys()}
    tables = {"cfr_eq": eq, "tight": table_of(tight), "aggressive": table_of(aggr), "random": uniform}
    # static exploiters (each is a best response to one archetype, both seats)
    def br_table(target):
        out = {}
        for seat in (0, 1):
            _, pol = best_response(GAME, target, seat)
            for k, a in pol.items():
                out[k] = {x: (1.0 if x == a else 0.0) for x in ("p", "b")}
        return out
    br_tight = br_table(tables["tight"])
    br_aggr = br_table(tables["aggressive"])
    cands = {k: tables[k] for k in ("cfr_eq", "tight", "aggressive", "random")}
    pool = {
        "random": lambda: RandomAgent(),
        "tight": lambda: ScriptedKuhnAgent("tight"),
        "aggressive": lambda: ScriptedKuhnAgent("aggressive"),
        "cfr_eq": lambda: PolicyTableAgent(eq, "Kuhn CFR+ 3000"),
        "br_vs_tight": lambda: PolicyTableAgent(br_tight, "static best response to tight"),
        "br_vs_aggressive": lambda: PolicyTableAgent(br_aggr, "static best response to aggressive"),
        "bayes": lambda: BayesArchetypeAgent(GAME, cands, decay=1.0, refresh=20, name="bayes"),
        "bayes_recency": lambda: BayesArchetypeAgent(GAME, cands, decay=0.97, refresh=20, name="bayes_recency"),
    }
    return pool, tables, eq


def match_job(args):
    na, nb, hands, seed = args
    pool, _, _ = build_pool()
    r = play_match(GAME, pool[na](), pool[nb](), hands, seed=seed)
    return na, nb, seed, r.mean_utility_a, r.se


def nash_of_matrix(M):
    """Row player's maximin mixed strategy of a symmetric zero-sum matrix game."""
    n = M.shape[0]
    # maximize v s.t. M^T x >= v, sum x = 1, x >= 0  -> minimize -v
    c = np.zeros(n + 1)
    c[-1] = -1
    A_ub = np.hstack([-M.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)])
    return res.x[:n], res.x[-1]


def fingerprint(agent_factory, opp_factory, hands=3000, seed=0):
    """Behavioral summary from play (seat-swapped): bet/call frequencies by card."""
    stats = {c: {"open_bet": [0, 0], "call_vs_bet": [0, 0]} for c in range(3)}
    a, b = agent_factory(), opp_factory()
    a.reset(); b.reset()
    rng = random.Random(seed)

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def choose_action(self, key, legal, r):
            act = self.inner.choose_action(key, legal, r)
            c, h = int(key[0]), key[1:]
            if h in ("", "p"):
                stats[c]["open_bet"][0] += act == "b"
                stats[c]["open_bet"][1] += 1
            else:
                stats[c]["call_vs_bet"][0] += act == "b"
                stats[c]["call_vs_bet"][1] += 1
            return act

        def observe_episode(self, *x):
            f = getattr(self.inner, "observe_episode", None)
            if f:
                f(*x)

        def reset(self):
            pass

        def metadata(self):
            return {}
    spy = Spy(a)
    for h in range(hands):
        seats = (spy, b) if h % 2 == 0 else (b, spy)
        play_episode(GAME, seats, rng)
    out = {}
    names = {0: "J", 1: "Q", 2: "K"}
    for c, d in stats.items():
        out[names[c]] = {k: (v[0] / v[1] if v[1] else None) for k, v in d.items()}
    return out


def regime_job(args):
    agent_name, seed, hands, switch_at, window = args
    pool, _, _ = build_pool()
    hero = pool[agent_name]()
    opp = SwitchingAgent(ScriptedKuhnAgent("tight"), ScriptedKuhnAgent("aggressive"), switch_at)
    hero.reset(); opp.reset()
    rng = random.Random(seed)
    utils = []
    for h in range(hands):
        if h % 2 == 0:
            u = play_episode(GAME, (hero, opp), rng)
        else:
            u = -play_episode(GAME, (opp, hero), rng)
        utils.append(u)
    u = np.array(utils)
    return {"agent": agent_name, "seed": seed,
            "window_means": [float(u[i:i + window].mean()) for i in range(0, hands, window)],
            "before": float(u[:switch_at].mean()), "after": float(u[switch_at:].mean()),
            "after_first_200": float(u[switch_at:switch_at + 200].mean())}


def calibration_job(args):
    true_name, seed, hands = args
    pool, tables, _ = build_pool()
    hero = pool["bayes"]()
    opp = pool[true_name]()
    hero.reset(); opp.reset()
    rng = random.Random(seed)
    trace = []
    for h in range(hands):
        seats = (hero, opp) if h % 2 == 0 else (opp, hero)
        play_episode(GAME, seats, rng)
        if h + 1 in (5, 10, 20, 50, 100, 200):
            trace.append({"hands": h + 1, "posterior_true": hero.posterior().get(true_name)})
    return {"true": true_name, "seed": seed, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    pool, tables, eq = build_pool()
    names = list(pool)
    hands = 300 if a.quick else 2000
    seeds = [0, 1] if a.quick else [0, 1, 2, 3, 4]
    jobs = [(x, y, hands, s * 7919 + i) for i, (x, y) in enumerate(itertools.combinations(names, 2)) for s in seeds]
    with Pool(a.workers) as p:
        res = p.map(match_job, jobs, chunksize=1)
    n = len(names)
    M = np.zeros((n, n))
    SE = np.zeros((n, n))
    per = {}
    for na, nb, seed, mean, se in res:
        per.setdefault((na, nb), []).append((mean, se))
    for (na, nb), vals in per.items():
        i, j = names.index(na), names.index(nb)
        m = float(np.mean([v[0] for v in vals]))
        se = float(np.sqrt(np.sum([v[1] ** 2 for v in vals])) / len(vals))
        M[i, j], M[j, i] = m, -m
        SE[i, j] = SE[j, i] = se
    sig = (np.abs(M) > 1.96 * SE) & (SE > 0)
    cycles = []
    for i, j, k in itertools.permutations(range(n), 3):
        if i < j and i < k and sig[i, j] and sig[j, k] and sig[k, i] and M[i, j] > 0 and M[j, k] > 0 and M[k, i] > 0:
            cycles.append([names[i], names[j], names[k]])
    x, v = nash_of_matrix(M)
    support = {names[i]: float(x[i]) for i in range(n) if x[i] > 1e-6}
    doc = {"format": "pokeralpha.phase2.agents/v1", "game": "Kuhn", "agents": {k: pool[k]().metadata() for k in names},
           "payoff_matrix": {"agents": names, "mean": M.round(5).tolist(), "se": SE.round(5).tolist(),
                             "units": "chips per hand to the row agent, seat-swapped", "hands_per_match": hands,
                             "match_seeds": len(seeds)},
           "non_transitivity": {"significant_3_cycles": cycles, "meta_game_nash_support": support, "meta_game_value": float(v),
                                "definition": "cycle = A beats B beats C beats A, each with |mean| > 1.96 SE"}}
    print("matrix rows:", names)
    for i in range(n):
        print(f"{names[i]:18s}", " ".join(f"{M[i, j]:+.3f}" for j in range(n)))
    print("cycles:", cycles[:5], "nash support:", support)
    # fingerprints vs a common opponent (random)
    doc["fingerprints_vs_random"] = {k: fingerprint(pool[k], pool["random"], hands=600 if a.quick else 3000) for k in names}
    # regime shift
    rj = [(ag, s, 1000 if a.quick else 3000, 500 if a.quick else 1500, 100) for ag in ("cfr_eq", "bayes", "bayes_recency", "br_vs_tight")
          for s in range(2 if a.quick else 8)]
    with Pool(a.workers) as p:
        reg = p.map(regime_job, rj, chunksize=1)
    rsum = {}
    for ag in ("cfr_eq", "bayes", "bayes_recency", "br_vs_tight"):
        rows = [r for r in reg if r["agent"] == ag]
        rsum[ag] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                        "se": float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows)))}
                    for k in ("before", "after", "after_first_200")}
        rsum[ag]["window_means"] = np.mean([r["window_means"] for r in rows], axis=0).round(4).tolist()
    doc["regime_shift"] = {"opponent": "tight then aggressive", "switch_at_hand": rj[0][3], "hands": rj[0][2],
                           "seeds": len({r["seed"] for r in reg}), "summary": rsum}
    print("regime:", {k: (round(v["before"]["mean"], 3), round(v["after"]["mean"], 3), round(v["after_first_200"]["mean"], 3)) for k, v in rsum.items()})
    # posterior calibration
    cj = [(t, s, 200) for t in ("cfr_eq", "tight", "aggressive", "random") for s in range(5 if a.quick else 30)]
    with Pool(a.workers) as p:
        cal = p.map(calibration_job, cj, chunksize=1)
    csum = {}
    for t in ("cfr_eq", "tight", "aggressive", "random"):
        rows = [c for c in cal if c["true"] == t]
        csum[t] = {str(pt["hands"]): float(np.mean([r["trace"][i]["posterior_true"] for r in rows]))
                   for i, pt in enumerate(rows[0]["trace"])}
    doc["posterior_calibration"] = {"definition": "mean posterior mass on the TRUE archetype after N hands (well-specified case)",
                                    "by_true_archetype": csum}
    print("calibration:", csum)
    doc["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = OUT if not a.quick else OUT.with_name("agents_quick.json")
    out.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", out, doc["seconds"], "s")


if __name__ == "__main__":
    main()
