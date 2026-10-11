"""Phase II Program 5: abstraction science on exact domains.

5A capacity curve: river subgames, uniform strength cells K in a grid vs
   error-guided refinement path; exact exploitability, infosets, table bytes,
   solve throughput -> capacity/error frontier.
5B action abstraction: river subgames, bet menus from coarse to fine. The
   exact cost of a restricted menu = value a player loses when restricted to
   menu M while the opponent keeps the finest menu (both solved exactly);
   tree growth and throughput for symmetric menus.
5C perfect vs imperfect recall: Leduc with history-forgetting and card
   abstractions; each abstract game is solved with CFR+ and the lifted
   strategy is evaluated by exact exploitability in real Leduc; convergence
   curves expose non-convergence under imperfect recall.
5D collision dispersion: does within-cell disagreement of the RAW
   equilibrium policies predict the cell's measured error (best-response
   gain) in the abstract solution?

    python experiments/phase2_abstraction_science.py [--quick] [--parts A,B,C,D]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_adaptive_abstraction import CONFIRM_BOARDS, sample_range  # noqa: E402
from phase33_reduced_games import SUBGAMES  # noqa: E402

from poker_alpha.games.base import Game  # noqa: E402
from poker_alpha.games.leduc import LeducPoker  # noqa: E402
from poker_alpha.research.refinement import RiverRefinementProblem, refine  # noqa: E402
from poker_alpha.research.river_solver import RiverRangeSolver, equal_count_cells  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import exploitability  # noqa: E402

OUTDIR = ROOT / "results" / "phase2"
BOARDS = list(SUBGAMES) + list(CONFIRM_BOARDS)


def table_bytes(solver):
    return int(sum(2 * r.nbytes for r in solver.regret if r is not None))


# ---------------------------------------------------------------- 5A -------
def capacity_instance(args):
    name, board, b0, b1, seed, n_hands, ks, iters = args
    rng = np.random.default_rng(seed)
    r0, r1 = sample_range(board, b0, n_hands, rng), sample_range(board, b1, n_hands, rng)
    s = RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    prob = RiverRefinementProblem(s, iters)
    pts = []
    for k in ks:
        kk = [min(k, s.n[0]), min(k, s.n[1])]
        t = time.perf_counter()
        sv = prob.solve(prob.uniform(kk))
        pts.append({"k": k, "exploitability": sv.exploitability, "infosets": sv.num_infosets,
                    "table_bytes": table_bytes(s), "solve_seconds": time.perf_counter() - t,
                    "iterations_per_second": iters / sv.seconds})
    r = refine(prob, prob.uniform([2, 2]), splits=2 * ks[-1] - 4 if ks[-1] < 12 else 20, signal="br_gain")
    path = [{"infosets": t["num_infosets"], "exploitability": t["exploitability_after"]} for t in r["trace"]]
    return {"name": name, "seed": seed, "n": s.n, "uniform": pts, "adaptive_path": path}


# ---------------------------------------------------------------- 5B -------
MENUS = {"A_pot": (1.0,), "B_half_pot": (0.5, 1.0), "C_third_3q_1.5": (0.33, 0.75, 1.5),
         "D_fine": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}


def action_instance(args):
    name, board, b0, b1, seed, n_hands, iters = args
    rng = np.random.default_rng(seed)
    r0, r1 = sample_range(board, b0, n_hands, rng), sample_range(board, b1, n_hands, rng)
    ref_menu = MENUS["D_fine"]

    def solve(m0, m1):
        s = RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=m0, raise_cap=2, bets_p1=m1)
        t = time.perf_counter()
        s.train(iters)
        secs = time.perf_counter() - t
        ev = s.evaluate()
        return {"value_p0": ev["ev_p0"], "exploitability_in_own_game": ev["exploitability"],
                "decision_nodes": len(s.tree.decision_nodes), "infosets": s.num_infosets(),
                "seconds": secs, "iterations_per_second": iters / secs}
    ref = solve(ref_menu, ref_menu)
    out = {"name": name, "seed": seed, "reference": ref, "menus": {}}
    for mname, m in MENUS.items():
        sym = solve(m, m)
        p0r = solve(m, ref_menu)
        p1r = solve(ref_menu, m)
        out["menus"][mname] = {"symmetric": sym,
                               "p0_restricted_loss": ref["value_p0"] - p0r["value_p0"],
                               "p1_restricted_loss": p1r["value_p0"] - ref["value_p0"],
                               "solver_error_bound": max(ref["exploitability_in_own_game"], p0r["exploitability_in_own_game"],
                                                         p1r["exploitability_in_own_game"])}
    return out


# ---------------------------------------------------------------- 5C -------
class KeyedGame(Game):
    """Same tree as ``inner``; information-set keys from ``keyfn``."""

    def __init__(self, inner, keyfn, name):
        self.inner, self.keyfn, self.name = inner, keyfn, name

    def signature(self):
        return f"{self.inner.signature()}:abstraction={self.name}"

    def root(self):
        return self.inner.root()

    def is_chance(self, s):
        return self.inner.is_chance(s)

    def chance_outcomes(self, s):
        return self.inner.chance_outcomes(s)

    def is_terminal(self, s):
        return self.inner.is_terminal(s)

    def utility(self, s):
        return self.inner.utility(s)

    def current_player(self, s):
        return self.inner.current_player(s)

    def infoset_key(self, s):
        return self.keyfn(self.inner, s)

    def legal_actions(self, s):
        return self.inner.legal_actions(s)

    def next_state(self, s, a):
        return self.inner.next_state(s, a)


def _parts(g, s):
    k = g.infoset_key(s)
    rank_, pub, hist = k.split("|", 2)
    return rank_, pub, hist


def k_perfect(g, s):
    return g.infoset_key(s)


def k_forget_r1(g, s):
    rank_, pub, hist = _parts(g, s)
    if "/" in hist:
        return f"{rank_}|{pub}|*/{hist.split('/', 1)[1]}"
    return f"{rank_}|{pub}|{hist}"


def k_r1_raises_only(g, s):
    rank_, pub, hist = _parts(g, s)
    if "/" in hist:
        r1, r2 = hist.split("/", 1)
        return f"{rank_}|{pub}|R{r1.count('r')}/{r2}"
    return f"{rank_}|{pub}|{hist}"


def k_forget_own_card_r2(g, s):
    """Round 2 keeps only whether the private card pairs the board and
    whether it is above / below the public card (forgets exact rank)."""
    rank_, pub, hist = _parts(g, s)
    if pub == "?":
        return f"{rank_}|{pub}|{hist}"
    r, p = int(rank_), int(pub)
    rel = "P" if r == p else ("H" if r > p else "L")
    return f"{rel}|{pub}|{hist}"


def k_merge_jq_r1(g, s):
    rank_, pub, hist = _parts(g, s)
    if pub == "?":
        rank_ = "JQ" if rank_ in ("0", "1") else rank_
    return f"{rank_}|{pub}|{hist}"


def k_merge_jq_both(g, s):
    rank_, pub, hist = _parts(g, s)
    rank_ = "JQ" if rank_ in ("0", "1") else rank_
    return f"{rank_}|{pub}|{hist}"


LEDUC_ABS = {"perfect": (k_perfect, "perfect recall (identity)"),
             "merge_JQ_round1_only": (k_merge_jq_r1, "card abstraction in round 1, exact in round 2 (perfect recall)"),
             "merge_JQ_both_rounds": (k_merge_jq_both, "card abstraction in both rounds (perfect recall)"),
             "round1_raise_count_only": (k_r1_raises_only, "round-1 history compressed to its raise count (imperfect recall)"),
             "forget_round1_history": (k_forget_r1, "round-1 betting forgotten in round 2 (imperfect recall)"),
             "forget_own_rank_round2": (k_forget_own_card_r2, "round 2 keeps only pair/above/below the public card (imperfect recall)")}


def leduc_job(args):
    name, checkpoints = args
    real = LeducPoker()
    fn, _ = LEDUC_ABS[name]
    g = KeyedGame(real, fn, name)
    s = CFRPlusSolver(g)
    curve, done = [], 0
    t0 = time.perf_counter()
    for c in checkpoints:
        s.train(c - done)
        done = c
        secs = time.perf_counter() - t0
        abstract = s.average_strategy()
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
                lifted[rk] = abstract.get(fn(real, st), {a: 1 / len(real.legal_actions(st)) for a in real.legal_actions(st)})
            stack.extend(real.next_state(st, a) for a in real.legal_actions(st))
        curve.append({"iterations": c, "exploitability_in_leduc": exploitability(real, lifted), "train_seconds": secs})
    return {"abstraction": name, "description": LEDUC_ABS[name][1], "infosets": len(s.infosets),
            "table_bytes": int(sum(n.regret_sum.nbytes * 2 for n in s.infosets.values())), "curve": curve,
            "seconds_per_iteration": curve[-1]["train_seconds"] / checkpoints[-1]}


# ---------------------------------------------------------------- 5D -------
def collision_instance(args):
    name, board, b0, b1, seed, n_hands, k, iters = args
    rng = np.random.default_rng(seed)
    r0, r1 = sample_range(board, b0, n_hands, rng), sample_range(board, b1, n_hands, rng)
    raw = RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2)
    raw.train(2 * iters)
    eq = raw.lifted(raw.average_strategy())
    reach = _hand_node_reach(raw, eq)
    prob = RiverRefinementProblem(RiverRangeSolver(board, r0, r1, pot=10.0, stack=20.0, bets=(0.5, 1.0), raise_cap=2), iters)
    cells = prob.uniform([k, k])
    sv = prob.solve(cells)
    rows = []
    for p in (0, 1):
        for c in range(k):
            members = np.flatnonzero(cells[p] == c)
            disp, wsum = 0.0, 0.0
            for i in raw.tree.decision_nodes_of(p):
                w = reach[i][members]
                if w.sum() <= 0:
                    continue
                pol = eq[i][members]
                mean = (w[:, None] * pol).sum(0) / w.sum()
                disp += float((w * 0.5 * np.abs(pol - mean).sum(1)).sum())
                wsum += float(w.sum())
            rows.append({"player": p, "cell": c, "policy_dispersion": disp / wsum if wsum > 0 else 0.0,
                         "equity_spread": float(prob.equity[p][members].max() - prob.equity[p][members].min()),
                         "br_gain": float(sv.signals["br_gain"][p][c]),
                         "shared_action_regret": float(sv.signals["shared_action_regret"][p][c])})
    return {"name": name, "seed": seed, "k": k, "cells": rows}


def _hand_node_reach(s, lifted):
    """Per-hand total reach (chance-weighted, both players) of each node, for
    the owner of the node: chance weight of the hand x own reach x mean opp reach."""
    out = {}
    T = s.tree

    def walk(i, r0, r1):
        node = T.nodes[i]
        if node.player < 0:
            return
        p = node.player
        own = r0 if p == 0 else r1
        opp = r1 if p == 0 else r0
        out[i] = own * (s.P[p] @ opp)
        for a, c in enumerate(node.children):
            if p == 0:
                walk(c, r0 * lifted[i][:, a], r1)
            else:
                walk(c, r0, r1 * lifted[i][:, a])
    walk(0, np.ones(s.n[0]), np.ones(s.n[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", type=str, default="A,B,C,D")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    parts = a.parts.split(",")
    seeds = [0] if a.quick else [0, 1, 2]
    boards = BOARDS[:2] if a.quick else BOARDS
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sfx = "_quick" if a.quick else ""
    with Pool(a.workers) as pool:
        if "A" in parts:
            ks = [1, 2, 3, 4, 6] if a.quick else [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
            t0 = time.time()
            inst = pool.map(capacity_instance, [(n, b, b0, b1, 300 + s, 16 if a.quick else 48, ks, 100 if a.quick else 400)
                                                for n, b, b0, b1 in boards for s in seeds], chunksize=1)
            summ = {}
            for k in ks:
                e = [next(p for p in i["uniform"] if p["k"] == k) for i in inst]
                summ[str(k)] = {"exploitability_mean": float(np.mean([x["exploitability"] for x in e])),
                                "infosets_mean": float(np.mean([x["infosets"] for x in e])),
                                "table_bytes_mean": float(np.mean([x["table_bytes"] for x in e])),
                                "iterations_per_second_mean": float(np.mean([x["iterations_per_second"] for x in e]))}
            # adaptive vs uniform at matched infosets: interpolate uniform curve (log error) at adaptive points
            gains = []
            for i in inst:
                ux = np.array([p["infosets"] for p in i["uniform"]], float)
                uy = np.log(np.array([p["exploitability"] for p in i["uniform"]]))
                for pt in i["adaptive_path"]:
                    if ux.min() <= pt["infosets"] <= ux.max():
                        gains.append(float(np.exp(np.interp(pt["infosets"], ux, uy)) / pt["exploitability"]))
            doc = {"format": "pokeralpha.phase2.abstraction_capacity/v1", "domain": "river subgames (Program 1 family)",
                   "k_grid": ks, "seeds": seeds, "boards": [b[0] for b in boards], "summary_uniform": summ,
                   "adaptive_vs_uniform_error_ratio_at_equal_infosets": {
                       "median": float(np.median(gains)) if gains else None, "mean": float(np.mean(gains)) if gains else None,
                       "share_adaptive_better": float(np.mean(np.array(gains) > 1)) if gains else None, "points": len(gains),
                       "definition": "uniform error interpolated (log-linear in infosets) / adaptive error; > 1 = adaptive better"},
                   "instances": inst, "seconds": round(time.time() - t0, 1), "commit": commit,
                   "timing_note": f"{a.workers} instances ran concurrently"}
            (OUTDIR / f"abstraction_capacity{sfx}.json").write_text(json.dumps(doc, indent=1, default=float))
            print("A", {k: round(v["exploitability_mean"], 4) for k, v in summ.items()}, doc["adaptive_vs_uniform_error_ratio_at_equal_infosets"], flush=True)
        if "B" in parts:
            t0 = time.time()
            inst = pool.map(action_instance, [(n, b, b0, b1, 400 + s, 12 if a.quick else 32, 100 if a.quick else 600)
                                              for n, b, b0, b1 in boards for s in seeds], chunksize=1)
            summ = {}
            for m in MENUS:
                rows = [i["menus"][m] for i in inst]
                summ[m] = {"bets": MENUS[m],
                           "p0_restricted_loss_mean": float(np.mean([r["p0_restricted_loss"] for r in rows])),
                           "p1_restricted_loss_mean": float(np.mean([r["p1_restricted_loss"] for r in rows])),
                           "p0_restricted_loss_max": float(np.max([r["p0_restricted_loss"] for r in rows])),
                           "p1_restricted_loss_max": float(np.max([r["p1_restricted_loss"] for r in rows])),
                           "decision_nodes": float(np.mean([r["symmetric"]["decision_nodes"] for r in rows])),
                           "infosets_mean": float(np.mean([r["symmetric"]["infosets"] for r in rows])),
                           "iterations_per_second_mean": float(np.mean([r["symmetric"]["iterations_per_second"] for r in rows])),
                           "solver_error_bound_max": float(np.max([r["solver_error_bound"] for r in rows]))}
            doc = {"format": "pokeralpha.phase2.action_abstraction/v1",
                   "definition": "restricted loss = exact value a player loses when limited to the menu while the opponent "
                                 "keeps the finest menu D (pot 10 chips, stack 20, raise cap 2); solver error bound reported",
                   "menus": MENUS, "summary": summ, "instances": inst, "seconds": round(time.time() - t0, 1), "commit": commit}
            (OUTDIR / f"action_abstraction{sfx}.json").write_text(json.dumps(doc, indent=1, default=float))
            print("B", {m: (round(v["p0_restricted_loss_mean"], 4), round(v["p1_restricted_loss_mean"], 4), v["decision_nodes"]) for m, v in summ.items()}, flush=True)
        if "C" in parts:
            t0 = time.time()
            cps = [10, 30, 100] if a.quick else [10, 30, 100, 300, 1000, 3000]
            res = pool.map(leduc_job, [(n, cps) for n in LEDUC_ABS], chunksize=1)
            doc = {"format": "pokeralpha.phase2.recall_ablation/v1", "game": "Leduc", "solver": "CFR+ on the abstract game",
                   "metric": "exact exploitability of the lifted strategy in real Leduc (chips)", "results": res,
                   "seconds": round(time.time() - t0, 1), "commit": commit}
            (OUTDIR / f"recall_ablation{sfx}.json").write_text(json.dumps(doc, indent=1, default=float))
            for r in res:
                print("C", r["abstraction"], r["infosets"], [round(c["exploitability_in_leduc"], 4) for c in r["curve"]], flush=True)
        if "D" in parts:
            t0 = time.time()
            inst = pool.map(collision_instance, [(n, b, b0, b1, 500 + s, 16 if a.quick else 48, 4, 100 if a.quick else 400)
                                                 for n, b, b0, b1 in boards for s in seeds], chunksize=1)
            cells = [c for i in inst for c in i["cells"]]
            br = np.array([c["br_gain"] for c in cells])
            out = {}
            for sig in ("policy_dispersion", "equity_spread", "shared_action_regret"):
                x = np.array([c[sig] for c in cells])
                out[sig] = {"spearman_vs_br_gain": float(spearmanr(x, br).statistic)}
                # top-1 agreement: is the cell with the max signal the max-br-gain cell, per instance
                hits = []
                for i in inst:
                    xs = [c[sig] for c in i["cells"]]
                    bs = [c["br_gain"] for c in i["cells"]]
                    hits.append(int(np.argmax(xs) == np.argmax(bs)))
                out[sig]["top_cell_agreement"] = float(np.mean(hits))
            doc = {"format": "pokeralpha.phase2.collision_dispersion/v1",
                   "question": "does within-cell heterogeneity of the RAW equilibrium predict the cell's measured error?",
                   "definitions": {"policy_dispersion": "reach-weighted mean TV of member hands' raw-equilibrium policies to "
                                                       "their weighted mean, over the player's nodes",
                                   "br_gain": "cell's exact best-response gain in the abstract solution (error decomposition)"},
                   "cells": len(cells), "k": 4, "signals": out, "instances": inst, "seconds": round(time.time() - t0, 1),
                   "commit": commit}
            (OUTDIR / f"collision_dispersion{sfx}.json").write_text(json.dumps(doc, indent=1, default=float))
            print("D", out, flush=True)


if __name__ == "__main__":
    main()
