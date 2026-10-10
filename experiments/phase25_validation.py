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


VISIT_TOL = 1e-6


def visit_counts(solver) -> np.ndarray:
    """Exact integer visit counts per infoset.

    External-sampling MCCFR adds the current strategy (a probability vector)
    to ``strategy_sum`` once per visit by the non-updating player, so
    ``strategy_sum.sum()`` equals the visit count up to floating-point
    summation error (e.g. 0.9999999999999999). Earlier histograms compared
    ``v == 1`` on these floats and silently dropped such infosets from every
    bucket, which is why their percentages summed to ~93%. Here each total is
    checked to lie within ``VISIT_TOL`` of an integer before rounding; a
    violation would indicate a real bug and raises.
    """
    raw = np.array([n.strategy_sum.sum() for n in solver.infosets.values()],
                   dtype=np.float64)
    rounded = np.rint(raw)
    bad = np.abs(raw - rounded) > VISIT_TOL * np.maximum(1.0, rounded)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} visit totals are not near-integers")
    return rounded.astype(np.int64)


def visit_stats(v: np.ndarray, prev_infosets: int) -> dict:
    n = len(v)
    cats = {"0": v == 0, "1": v == 1, "2-5": (v >= 2) & (v <= 5),
            "6-20": (v >= 6) & (v <= 20), ">20": v > 20}
    counts = {k: int(m.sum()) for k, m in cats.items()}
    assert sum(counts.values()) == n
    return {
        "infosets": n,
        "counts": counts,
        "percent": {k: 100.0 * c / n for k, c in counts.items()},
        "max": int(v.max()), "median": float(np.median(v)), "mean": float(v.mean()),
        "max_visit_non_integer_totals": 0,
        # discovery vs learning (see docs/validation.md for definitions)
        "fraction_newly_discovered": (n - prev_infosets) / n,
        "fraction_revisited_ge2": float(np.mean(v >= 2)),
        "fraction_trained_ge5": float(np.mean(v >= 5)),
        "fraction_trained_ge10": float(np.mean(v >= 10)),
        "fraction_trained_ge20": float(np.mean(v >= 20)),
        "visit_share_in_infosets_ge5": float(v[v >= 5].sum() / max(v.sum(), 1)),
    }


def top_visited(solver, n: int):
    items = sorted(((float(node.strategy_sum.sum()), k) for k, node in solver.infosets.items()),
                   key=lambda x: (-x[0], x[1]))[:n]
    return {k: (v, solver.infosets[k].average_strategy().copy()) for v, k in items}


def policy_record(p) -> dict:
    if not p.visited:
        return {"key": p.key, "visits": 0, "status": "UNVISITED", "policy": None}
    return {"key": p.key, "visits": int(round(p.visits)), "status": "visited",
            "policy": dict(zip(p.actions, [round(x, 4) for x in p.probs])),
            "l1_from_uniform": round(p.l1_from_uniform, 4)}


def part_c(runs_dir: Path, top_n: int = 2000) -> dict:
    game = make_game()
    spots = canonical_spots()
    runs = {}
    for f in sorted(runs_dir.glob("run_seed*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        runs[rows[0]["seed"]] = rows
    result = {"definitions": {
        "visit": "one arrival at the infoset as the non-updating player in external-sampling "
                 "MCCFR (each adds one probability vector to strategy_sum); infosets reached "
                 "only as the updating player have 0 visits",
        "newly_discovered": "infoset absent at the previous checkpoint",
        "revisited": "visit count >= 2",
        "meaningfully_trained": "reported at three fixed thresholds: >= 5, >= 10, >= 20 visits; "
                                "none is privileged",
        "l1": "sum of absolute probability differences; maximum possible 2.0"},
        "runs": {}, "seed_disagreement": {}}
    at_1000 = {}
    for seed, rows in sorted(runs.items()):
        per, prev_n, prev_pols = [], 0, None
        for r in rows:
            solver = _load(Path(r["checkpoint"]))
            v = visit_counts(solver)
            pols = spot_policies(game, solver.infosets, spots)
            top = top_visited(solver, top_n)
            row = {
                "iterations": r["iterations"],
                "wall_clock_seconds_cumulative": r["total_seconds"],
                "segment_seconds": r["segment_seconds"],
                "iterations_per_second_segment": r["iters_per_sec_segment"],
                "new_infosets_since_previous": r["infosets"] - prev_n,
                "checkpoint_bytes": Path(r["checkpoint"]).stat().st_size,
                "infoset_memory_estimate_bytes": int(r["memory_mb_estimate"] * 1e6),
                "max_rss_mb": r["max_rss_mb"],
                "entropy_bits_visit_weighted": r["entropy_bits_visit_weighted"],
                "visits": visit_stats(v, prev_n),
                f"top{top_n}_most_visited_min_visits": int(round(min(x for x, _ in top.values()))),
                f"top{top_n}_share_far_from_uniform_l1_gt_0.5": float(np.mean(
                    [np.abs(p - 1.0 / len(p)).sum() > 0.5 for _, p in top.values()])),
                "canonical": {},
            }
            assert row["visits"]["infosets"] == r["infosets"]
            for i, p in enumerate(pols):
                rec = policy_record(p)
                if prev_pols is not None and p.visited and prev_pols[i].visited:
                    rec["l1_vs_previous_checkpoint"] = round(l1(prev_pols[i], p), 4)
                row["canonical"][p.spot] = rec
            per.append(row)
            if r["iterations"] == 1000:
                at_1000[seed] = (top, pols)
            prev_n, prev_pols = r["infosets"], pols
            print(f"[C] seed {seed} it {r['iterations']}: {row['visits']['counts']}", flush=True)
            del solver
        result["runs"][str(seed)] = per
    if len(at_1000) >= 2:
        seeds = sorted(at_1000)
        spot_rows = {}
        for i, sp in enumerate(spots):
            visited = [s for s in seeds if at_1000[s][1][i].visited]
            entry = {"visited_by": visited,
                     "category": ("all" if len(visited) == len(seeds) else
                                  "none" if not visited else "some"),
                     "pairwise_l1": {}}
            for a, b in itertools.combinations(seeds, 2):
                pa, pb = at_1000[a][1][i], at_1000[b][1][i]
                entry["pairwise_l1"][f"{a}v{b}"] = (round(l1(pa, pb), 4)
                                                    if pa.visited and pb.visited else None)
            spot_rows[sp.name] = entry
        pairs = {}
        for a, b in itertools.combinations(seeds, 2):
            ta, tb = at_1000[a][0], at_1000[b][0]
            common = sorted(set(ta) & set(tb))
            d = [float(np.abs(ta[k][1] - tb[k][1]).sum()) for k in common]
            spot_d = [e["pairwise_l1"][f"{a}v{b}"] for e in spot_rows.values()
                      if e["pairwise_l1"][f"{a}v{b}"] is not None]
            pairs[f"seed {a} vs seed {b}"] = {
                "canonical_spots_visited_by_both": len(spot_d),
                "canonical_mean_l1": float(np.mean(spot_d)) if spot_d else None,
                f"top{top_n}_overlap_fraction": len(common) / top_n,
                f"top{top_n}_common_mean_l1": float(np.mean(d)) if d else None,
                f"top{top_n}_common_median_l1": float(np.median(d)) if d else None,
            }
        result["seed_disagreement"] = {
            "iterations": 1000, "seeds": seeds, "max_possible_l1": 2.0,
            "note": "unvisited (uniform-fallback) infosets are excluded from every L1 average",
            "canonical_spots": spot_rows,
            "spot_counts": {c: sum(1 for e in spot_rows.values() if e["category"] == c)
                            for c in ("all", "some", "none")},
            "pairs": pairs}
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
