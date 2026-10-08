"""Phase 25: assemble the Hold'em platform validation artifact.

Runs (A) the encoder-independent compression measurement, (B) the
perfect-recall collision audit, (C) the analysis of the controlled MCCFR
runs produced by ``holdem_mccfr_validation.py`` (re-reading every checkpoint:
visit distribution, canonical-spot policies over time, seed-to-seed
disagreement), (D) the decision-invariant test suite, and records the
observer status. Writes ``results/validation/holdem_platform_validation.json``.

Usage
-----
    python experiments/phase25_validation.py --runs-dir RUNS --hands 2000
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from holdem_mccfr_validation import BETS, make_game  # noqa: E402

from poker_alpha.abstraction import (HoldemBucketEncoder,  # noqa: E402
                                     RawHoldemEncoder, ToyHoldemEncoder)
from poker_alpha.games import HoldemGame  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402
from poker_alpha.validation.abstraction_audit import (  # noqa: E402
    audit_perfect_recall, generate_corpus, generate_line_corpus,
    measure_compression)

FIXED_LINES = {
    "preflop BTN unopened": ("",),
    "flop BB first after BTN raise/call": ("b75c", ""),
    "turn BB first after flop check-check": ("b75c", "cc", ""),
    "river BB first after check-check twice": ("b75c", "cc", "cc", ""),
    "flop BB facing c-bet": ("b75c", "cb75"),
}
from poker_alpha.validation.canonical_spots import (canonical_spots,  # noqa: E402
                                                    l1, spot_policies)


def part_a_b(hands: int, seed: int) -> dict:
    out = {}
    for label, bets in (("training_menu_33_75_150", dict(BETS)),
                        ("default_menu_50_100_200", None)):
        game = HoldemGame(starting_stack=100.0, bet_fractions=bets)
        t = time.perf_counter()
        corpus = generate_corpus(game, hands, seed)
        encs = {"toy": ToyHoldemEncoder(), "bucket": HoldemBucketEncoder()}
        comp = measure_compression(game, corpus, encs)
        history_bound = len({(game.current_player(s), s.streets) for s in corpus})
        hist_by_street = {}
        for s in corpus:
            st = ("preflop", "flop", "turn", "river")[len(s.streets) - 1]
            hist_by_street.setdefault(st, set()).add((game.current_player(s), s.streets))
        audits = {n: audit_perfect_recall(game, corpus, e, n).to_dict()
                  for n, e in (("raw", RawHoldemEncoder()), ("toy", encs["toy"]),
                               ("bucket", encs["bucket"]))}
        d = comp.to_dict()
        d["history_lower_bound"] = history_bound
        d["history_lower_bound_by_street"] = {k: len(v) for k, v in hist_by_street.items()}
        d["corpus"] = {"hands": hands, "seed": seed, "policy":
                       "random legal: fold 0.15 / check-call 0.5 / bets share the rest",
                       "bet_menu": game.signature()}
        d["perfect_recall_audit"] = audits
        if bets is not None:
            lines = {}
            for name, line in FIXED_LINES.items():
                lc = generate_line_corpus(game, line, hands, seed + 1)
                lcomp = measure_compression(game, lc, encs)
                lines[name] = {"line": "/".join(line), "states": lcomp.states,
                               "raw_keys": lcomp.raw_keys, "keys": lcomp.keys,
                               "ratios": lcomp.ratios,
                               "invariant_violations": lcomp.invariant_violations,
                               "bucket_recall_violations": audit_perfect_recall(
                                   game, lc, encs["bucket"], "bucket").violations}
                # Cardinality of the last street's own bucket label alone, vs the
                # full key (which also carries the preflop class and earlier
                # buckets for perfect recall).
                last_labels = {encs["bucket"].encode(game, x).split("|")[1].split("/")[-1]
                               for x in lc}
                lines[name]["distinct_current_street_bucket_labels"] = len(last_labels)
                print(f"[A] line {name}: {lcomp.states} states, raw {lcomp.raw_keys}, "
                      f"current-street labels {len(last_labels)}, "
                      f"keys {lcomp.keys}", flush=True)
            d["fixed_line_corpora"] = lines
        d["seconds"] = time.perf_counter() - t
        out[label] = d
        print(f"[A/B] {label}: states {comp.states} raw keys {comp.raw_keys} "
              f"keys {comp.keys} history bound {history_bound}", flush=True)
    return out


def _load(path: Path):
    return load_checkpoint(path, make_game())


def visit_hist(solver) -> dict:
    v = np.rint(np.array([n.strategy_sum.sum() for n in solver.infosets.values()]))
    return {"0": float(np.mean(v == 0)), "1": float(np.mean(v == 1)),
            "2-5": float(np.mean((v >= 2) & (v <= 5))),
            "6-20": float(np.mean((v >= 6) & (v <= 20))), ">20": float(np.mean(v > 20)),
            "max": float(v.max())}


def top_visited(solver, n: int):
    items = sorted(((float(node.strategy_sum.sum()), k) for k, node in solver.infosets.items()),
                   key=lambda x: (-x[0], x[1]))[:n]
    return {k: (v, solver.infosets[k].average_strategy().copy()) for v, k in items}


def part_c(runs_dir: Path, top_n: int = 2000) -> dict:
    game = make_game()
    spots = canonical_spots()
    runs = {}
    for f in sorted(runs_dir.glob("run_seed*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        runs[rows[0]["seed"]] = rows
    result = {"runs": {}, "canonical": {}, "seed_disagreement": {}}
    tops = {}
    for seed, rows in sorted(runs.items()):
        per = []
        prev_spot = None
        for r in rows:
            solver = _load(Path(r["checkpoint"]))
            pols = spot_policies(game, solver.infosets, spots)
            hist = visit_hist(solver)
            top = top_visited(solver, top_n)
            far = np.mean([np.abs(p - 1.0 / len(p)).sum() > 0.5 for _, p in top.values()])
            row = {k: r[k] for k in ("iterations", "segment_seconds", "total_seconds",
                                     "iters_per_sec_segment", "infosets", "new_infosets",
                                     "top_n_prev_mean_l1", "top_n_prev_median_l1",
                                     "entropy_bits_visit_weighted", "checkpoint_mb",
                                     "memory_mb_estimate", "max_rss_mb")}
            row["visit_fraction"] = hist
            row["new_infoset_share"] = r["new_infosets"] / r["infosets"]
            row[f"top{top_n}_min_visits"] = min(v for v, _ in top.values())
            row[f"top{top_n}_share_far_from_uniform(L1>0.5)"] = float(far)
            row["canonical"] = {p.spot: {"visits": p.visits, "key": p.key,
                                         "policy": dict(zip(p.actions, [round(x, 4) for x in p.probs])),
                                         "l1_from_uniform": round(p.l1_from_uniform, 4),
                                         "l1_vs_previous_checkpoint":
                                             None if prev_spot is None else l1(prev_spot[i], p)}
                                for i, p in enumerate(pols)}
            prev_spot = pols
            per.append(row)
            if r["iterations"] == 1000:
                tops[seed] = (top, pols)
            print(f"[C] seed {seed} it {r['iterations']}: infosets {r['infosets']} "
                  f"visits {hist}", flush=True)
            del solver
        result["runs"][str(seed)] = per
    if len(tops) >= 2:
        pair = {}
        spot_l1 = {sp.name: [] for sp in spots}
        overlap, top_l1 = [], []
        for a, b in itertools.combinations(sorted(tops), 2):
            (ta, pa), (tb, pb) = tops[a], tops[b]
            common = set(ta) & set(tb)
            overlap.append(len(common) / top_n)
            top_l1.extend(float(np.abs(ta[k][1] - tb[k][1]).sum()) for k in common)
            for x, y in zip(pa, pb):
                if x.visited and y.visited:
                    spot_l1[x.spot].append(l1(x, y))
        pair["iterations"] = 1000
        pair["seeds"] = sorted(tops)
        pair[f"top{top_n}_key_overlap_mean"] = float(np.mean(overlap))
        pair[f"top{top_n}_common_policy_l1_mean"] = float(np.mean(top_l1)) if top_l1 else None
        pair[f"top{top_n}_common_policy_l1_median"] = float(np.median(top_l1)) if top_l1 else None
        pair["canonical_spot_l1_mean_where_visited_in_both"] = {
            k: (float(np.mean(v)) if v else None) for k, v in spot_l1.items()}
        result["seed_disagreement"] = pair
    return result


def part_d() -> dict:
    out = ROOT / "results" / "validation" / "_invariants_junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "tests/test_decision_invariants.py", f"--junitxml={out}"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    import xml.etree.ElementTree as ET

    root = ET.parse(out).getroot()
    suite = root.find("testsuite")
    suite = root if suite is None else suite
    cases = [{"test": c.get("name"),
              "status": "failed" if c.find("failure") is not None or c.find("error") is not None
              else ("skipped" if c.find("skipped") is not None else "passed")}
             for c in suite.iter("testcase")]
    out.unlink()
    return {"command": " ".join(cmd[2:]), "returncode": proc.returncode,
            "summary": proc.stdout.strip().splitlines()[-1], "cases": cases}


def observer_status() -> dict:
    import csv

    syn = ROOT / "results" / "data" / "observer_synthetic_validation.csv"
    rows = list(csv.DictReader(syn.open())) if syn.exists() else []
    real = ROOT / "tests" / "fixtures" / "pokernow"
    imgs = [p for p in (real / "raw").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    anns = list((real / "annotations").glob("*.json"))
    return {"synthetic": rows,
            "synthetic_note": "rendered by poker_alpha.observer.synthetic; says nothing about real clients",
            "real_pokernow": {"screenshots": len(imgs), "annotations": len(anns),
                              "accuracy": None,
                              "status": "not validated: no real screenshots collected"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "holdem_platform_validation.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": "pokeralpha.validation/v1",
        "commit_base": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "numpy": np.__version__},
        "abstraction": part_a_b(args.hands, args.seed),
        "mccfr": part_c(args.runs_dir),
        "decision_invariants": part_d(),
        "observer": observer_status(),
        "exploitability": "not computed: exact Hold'em exploitability is infeasible here and no proxy is reported as exploitability",
    }
    args.out.write_text(json.dumps(doc, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
