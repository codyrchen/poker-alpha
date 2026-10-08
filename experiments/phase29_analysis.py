"""Phase 29: analyse large-scale training of the locked HU Hold'em config.

Inputs: runs of ``holdem_mccfr_validation.py --locked-config`` (JSONL rows +
checkpoints) for several seeds. Outputs ``results/validation/
holdem_training_v1.json`` with

* convergence proxies per seed and checkpoint (discovery vs revisitation,
  top-N policy movement, canonical-matrix movement, entropy);
* seed disagreement per checkpoint (canonical matrix, top-2000 infosets);
* the broad canonical matrix at the final checkpoint of every seed;
* strategic sanity checks (pass / fail / insufficient data, never hidden);
* duplicate seat-swapped cross-play (seeds vs seeds, final vs earlier
  checkpoints, vs uniform-random and calling-station baselines);
* a strategy artifact export (written outside the repo unless small).

No exploitability is computed; cross-play inside the abstract game is not an
exploitability bound.

Usage::

    python experiments/phase29_analysis.py --runs-dir RUNS --artifact OUT.npz
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase25_validation import top_visited, visit_counts, visit_stats  # noqa: E402

from poker_alpha.solver_config import PRIMARY_CONFIG  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402
from poker_alpha.solvers.strategy_artifact import export_solver  # noqa: E402
from poker_alpha.validation.canonical_matrix import (PREMIUM, TRASH,  # noqa: E402
                                                     canonical_matrix,
                                                     evaluate_matrix,
                                                     solver_lookup)
from poker_alpha.validation.crossplay import (calling_station,  # noqa: E402
                                              duplicate_match, table_policy,
                                              uniform_policy)

MIN_VISITS_SANITY = 5


def load_runs(runs_dir: Path):
    runs = {}
    for f in sorted(runs_dir.glob("locked_seed*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if rows:
            runs[rows[0]["seed"]] = rows
    return runs


def matrix_l1(a_rows, b_rows, min_visits=1.0):
    d = []
    for a, b in zip(a_rows, b_rows):
        if a["policy"] and b["policy"] and a["visits"] >= min_visits and b["visits"] >= min_visits:
            d.append(sum(abs(a["policy"][k] - b["policy"][k]) for k in a["policy"]))
    return {"spots": len(d), "mean_l1": float(np.mean(d)) if d else None,
            "median_l1": float(np.median(d)) if d else None}


# -- sanity checks -----------------------------------------------------------

def _rows(rows, **kw):
    out = []
    for r in rows:
        if r["status"] != "visited" or r["visits"] < MIN_VISITS_SANITY:
            continue
        ok = True
        for k, v in kw.items():
            val = r[k]
            if isinstance(v, (tuple, list, set)):
                ok &= val in v
            elif callable(v):
                ok &= bool(v(val))
            else:
                ok &= val == v
        if ok:
            out.append(r)
    return out


def _mean(rows, cls):
    return float(np.mean([r["classes"][cls] for r in rows])) if rows else None


def _check(name, lhs, rhs, op, n, detail):
    if lhs is None or (rhs is None and op not in ("<=c", ">=c")):
        return {"check": name, "result": "INSUFFICIENT DATA", "spots_used": n, "detail": detail}
    if op == ">":
        ok = lhs > rhs
    elif op == "<=c":
        ok = lhs <= rhs
    else:
        raise ValueError(op)
    return {"check": name, "result": "PASS" if ok else "FAIL", "lhs": round(lhs, 4),
            "rhs": rhs if isinstance(rhs, (int, float)) and op == "<=c" else round(rhs, 4),
            "spots_used": n, "detail": detail}


def sanity_checks(rows):
    out = []
    aa = _rows(rows, street="preflop", hand="AA")
    out.append(_check("AA never folds preflop (max fold prob)",
                      max((r["classes"]["fold"] for r in aa), default=None), 0.05, "<=c",
                      len(aa), "every preflop situation"))
    btn = "BTN unopened"
    f72 = _rows(rows, street="preflop", situation=btn, hand="72o")
    faa = _rows(rows, street="preflop", situation=btn, hand="AA")
    out.append(_check("BTN folds 72o more often than AA", _mean(f72, "fold"), _mean(faa, "fold"),
                      ">", len(f72) + len(faa), btn))
    pr = _rows(rows, street="preflop", situation=btn, hand=PREMIUM)
    tr = _rows(rows, street="preflop", situation=btn, hand=TRASH)
    out.append(_check("BTN raises premiums more often than trash", _mean(pr, "aggressive"),
                      _mean(tr, "aggressive"), ">", len(pr) + len(tr), f"{PREMIUM} vs {TRASH}"))
    vs = ("BB vs raise33", "BB vs raise75", "BB vs raise150")
    tr = _rows(rows, street="preflop", situation=vs, hand=TRASH)
    pr = _rows(rows, street="preflop", situation=vs, hand=PREMIUM)
    out.append(_check("BB folds trash more often than premiums vs a raise", _mean(tr, "fold"),
                      _mean(pr, "fold"), ">", len(tr) + len(pr), "raise 33/75/150"))
    tr = _rows(rows, street="preflop", situation="BB vs all-in", hand=TRASH)
    pr = _rows(rows, street="preflop", situation="BB vs all-in", hand=PREMIUM)
    out.append(_check("BB folds trash more often than premiums vs all-in", _mean(tr, "fold"),
                      _mean(pr, "fold"), ">", len(tr) + len(pr), ""))
    facing = lambda s: s.startswith("BB facing")  # noqa: E731
    air = _rows(rows, street="flop", situation=facing, hand="air")
    tpp = _rows(rows, street="flop", situation=facing, hand="top_pair_plus")
    out.append(_check("Facing a flop bet, air folds more often than top pair+",
                      _mean(air, "fold"), _mean(tpp, "fold"), ">", len(air) + len(tpp), ""))
    mon = _rows(rows, street="flop", situation=facing, hand="monster")
    out.append(_check("Facing a flop bet, monsters fold <= 10% (mean)", _mean(mon, "fold"), 0.10,
                      "<=c", len(mon), ""))
    weak = ("air", "weak_made")
    big = _rows(rows, street="flop", situation="BB facing bet150", hand=weak)
    small = _rows(rows, street="flop", situation="BB facing bet33", hand=weak)
    out.append(_check("Air / weak made hands fold more to 150% than to 33% bets",
                      _mean(big, "fold"), _mean(small, "fold"), ">", len(big) + len(small), ""))
    first = ("BB first", "BTN after check")
    val = _rows(rows, street="flop", situation=first, hand=("monster", "top_pair_plus"))
    wm = _rows(rows, street="flop", situation=first, hand="weak_made")
    out.append(_check("First to act on the flop, value hands bet more often than weak made hands",
                      _mean(val, "aggressive"), _mean(wm, "aggressive"), ">", len(val) + len(wm),
                      "weak made = bluff-catchers; air may legitimately bluff"))
    rv = _rows(rows, street="river", situation=facing, hand="monster")
    out.append(_check("River monsters facing a bet fold <= 5% (mean)", _mean(rv, "fold"), 0.05,
                      "<=c", len(rv), ""))
    return out


# -- cross-play ---------------------------------------------------------------

_POLICIES = {}


def _strategy_table(solver):
    return {k: (dict(zip(n.actions, n.average_strategy().tolist())), float(n.strategy_sum.sum()))
            for k, n in solver.infosets.items() if n.strategy_sum.sum() > 0}


def _match(job):
    name, a, b, deals, seed = job
    game = PRIMARY_CONFIG.build_game()
    pa, pb = _POLICIES[a](), _POLICIES[b]()
    t = time.perf_counter()
    r = duplicate_match(game, pa, pb, deals, seed).to_dict()
    r["seconds"] = time.perf_counter() - t
    for tag, p in (("a", pa), ("b", pb)):
        st = getattr(p, "stats", None)
        if st:
            tot = st["hits"] + st["misses"]
            r[f"{tag}_lookup_miss_rate"] = st["misses"] / tot if tot else None
    return name, a, b, r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--deals", type=int, default=200000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "holdem_training_v1.json")
    args = p.parse_args()
    runs = load_runs(args.runs_dir)
    game = PRIMARY_CONFIG.build_game()
    spots = canonical_matrix()
    seeds = sorted(runs)
    per_seed, mats, tops, tables = {}, {}, {}, {}
    for seed in seeds:
        out_rows, prev_n, prev_m, prev_it = [], 0, None, 0
        for r in runs[seed]:
            solver = load_checkpoint(Path(r["checkpoint"]), game)
            v = visit_counts(solver)
            m = evaluate_matrix(game, solver_lookup(solver), spots)
            it = r["iterations"]
            vis = [x for x in m if x["status"] == "visited"]
            row = {
                "iterations": it, "wall_clock_seconds": r["total_seconds"],
                "iterations_per_second_segment": r["iters_per_sec_segment"],
                "infosets": len(solver.infosets),
                "new_infosets_per_iteration": (len(solver.infosets) - prev_n) / (it - prev_it),
                "visits": visit_stats(v, prev_n),
                "top_n_prev_mean_l1": r.get("top_n_prev_mean_l1"),
                "top_n_prev_median_l1": r.get("top_n_prev_median_l1"),
                "entropy_bits_visit_weighted": r["entropy_bits_visit_weighted"],
                "checkpoint_mb": r["checkpoint_mb"],
                "matrix_visited": len(vis),
                "matrix_ge20": sum(1 for x in vis if x["visits"] >= 20),
                "matrix_l1_vs_previous": matrix_l1(prev_m, m) if prev_m else None,
            }
            out_rows.append(row)
            mats.setdefault(it, {})[seed] = m
            tops.setdefault(it, {})[seed] = top_visited(solver, 2000)
            print(f"[{seed}] it {it}: infosets {row['infosets']} ge5 "
                  f"{row['visits']['fraction_trained_ge5']:.3f} matrix {len(vis)}/{len(m)}", flush=True)
            if r is runs[seed][-1] or it in (1000, 10000, 100000):
                tables[(seed, it)] = _strategy_table(solver)
            if seed == seeds[0] and r is runs[seed][-1]:
                art = export_solver(solver, args.artifact, meta={
                    "seed": seed, "iterations": it, "source_checkpoint": Path(r["checkpoint"]).name,
                    "trained_with": "external-sampling MCCFR",
                    "recall": "IMPERFECT RECALL - no standard CFR equilibrium guarantee"})
            prev_n, prev_m, prev_it = len(solver.infosets), m, it
            del solver
        per_seed[str(seed)] = out_rows
    final_it = min(runs[s][-1]["iterations"] for s in seeds)

    disagreement = {}
    for it in sorted(mats):
        if len(mats[it]) < 2:
            continue
        pairs = {}
        for a, b in itertools.combinations(sorted(mats[it]), 2):
            ta, tb = tops[it][a], tops[it][b]
            common = sorted(set(ta) & set(tb))
            d = [float(np.abs(ta[k][1] - tb[k][1]).sum()) for k in common]
            pairs[f"seed {a} vs seed {b}"] = {
                "matrix_both_visited": matrix_l1(mats[it][a], mats[it][b]),
                "matrix_both_ge20": matrix_l1(mats[it][a], mats[it][b], 20.0),
                "top2000_overlap": len(common) / 2000,
                "top2000_common_mean_l1": float(np.mean(d)) if d else None}
        disagreement[str(it)] = {"max_possible_l1": 2.0, "pairs": pairs}

    sanity = {str(s): sanity_checks(mats[final_it][s]) for s in seeds if s in mats[final_it]}

    # Cross-play (fork shares the strategy tables with workers).
    for (s, it), tab in tables.items():
        _POLICIES[f"seed{s}@{it}"] = (lambda t=tab: table_policy(lambda k: t.get(k)))
    _POLICIES["uniform"] = lambda: uniform_policy
    _POLICIES["calling_station"] = lambda: calling_station
    s0, s1, s2 = (seeds + [None, None])[:3]
    F = final_it
    jobs = []
    if s1 is not None:
        jobs.append(("seed vs seed (final)", f"seed{s0}@{F}", f"seed{s1}@{F}"))
    if s2 is not None:
        jobs.append(("seed vs seed (final)", f"seed{s1}@{F}", f"seed{s2}@{F}"))
        jobs.append(("seed vs seed (final)", f"seed{s0}@{F}", f"seed{s2}@{F}"))
    if s1 is not None and (s1, 1000) in tables:
        jobs.append(("final vs 1k (other seed)", f"seed{s0}@{F}", f"seed{s1}@1000"))
    if s1 is not None and (s1, 10000) in tables and F > 10000:
        jobs.append(("final vs 10k (other seed)", f"seed{s0}@{F}", f"seed{s1}@10000"))
    if s1 is not None and (s1, 100000) in tables and F > 100000:
        jobs.append(("final vs 100k (other seed)", f"seed{s0}@{F}", f"seed{s1}@100000"))
        jobs.append(("final vs 100k (same seed)", f"seed{s0}@{F}", f"seed{s0}@100000"))
    jobs.append(("final vs uniform random", f"seed{s0}@{F}", "uniform"))
    jobs.append(("final vs calling station", f"seed{s0}@{F}", "calling_station"))
    jobs.append(("control: uniform vs uniform", "uniform", "uniform"))
    jobs = [(n, a, b, args.deals, 7 + i) for i, (n, a, b) in enumerate(jobs)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        cross = [dict(match=n, a=a, b=b, **r) for n, a, b, r in pool.map(_match, jobs)]
    for c in cross:
        print(f"{c['match']}: {c['a']} vs {c['b']}: {c['bb_per_100']:.1f} bb/100 "
              f"CI {c['ci95_bb_per_100'][0]:.1f}..{c['ci95_bb_per_100'][1]:.1f}", flush=True)

    doc = {
        "format": "pokeralpha.holdem_training/v1",
        "config": PRIMARY_CONFIG.to_dict(),
        "config_signature": PRIMARY_CONFIG.signature(),
        "recall": "IMPERFECT RECALL - no standard CFR equilibrium guarantee",
        "seeds": seeds, "final_iterations": final_it,
        "convergence_proxies": per_seed,
        "seed_disagreement": disagreement,
        "canonical_matrix": {"spots": len(spots), "final": {str(s): mats[final_it][s] for s in seeds}},
        "sanity_checks": sanity,
        "crossplay": {"deals_per_match": args.deals,
                      "note": "duplicate deals, seat-swapped; inside the abstract game only; "
                              "unvisited infosets play uniform; NOT an exploitability measure",
                      "matches": cross},
        "artifact": {"path": str(args.artifact), "bytes": art.stat().st_size,
                     "seed": seeds[0], "iterations": runs[seeds[0]][-1]["iterations"]},
        "exploitability": "not computed",
    }
    args.out.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
