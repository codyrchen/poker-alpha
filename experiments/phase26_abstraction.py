"""Phase 26: compare scalable Hold'em abstractions -> results/validation/abstraction_v2.json.

* same frozen Phase-25 corpus (2,000 random-policy hands, seed 0, 33/75/150%
  menu) and the same fixed action lines, through every candidate encoder:
  keys and compression by street;
* within-key quality: equity mean/std/range vs a uniform random hand
  (offline reference only), hand-category / draw / SPR / legal-action mixing;
* perfect-recall audits (Phase-25 inspector);
* training comparison from runs of ``holdem_mccfr_validation.py`` (identical
  seeds and budgets per encoder): infosets, new infosets per iteration, visit
  distribution, canonical-state visits and policies, policy movement, seed
  disagreement, runtime.

Usage
-----
    python experiments/phase26_abstraction.py --runs-dir P26 --bucket-runs-dir P25
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from holdem_mccfr_validation import BETS, ENCODER_NAMES, make_encoder  # noqa: E402
from phase25_validation import (FIXED_LINES, policy_record,  # noqa: E402
                                top_visited, visit_counts, visit_stats)

from poker_alpha.abstraction import RawHoldemEncoder  # noqa: E402
from poker_alpha.games import HoldemGame  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402
from poker_alpha.validation.abstraction_audit import (  # noqa: E402
    audit_perfect_recall, generate_corpus, generate_line_corpus,
    measure_compression, measure_quality, state_facts)
from poker_alpha.validation.canonical_spots import (canonical_spots, l1,  # noqa: E402
                                                    spot_policies)

CANDIDATES = ("bucket", "transition", "transition_abstract", "compact", "compact_exact")


def game_for(encoder: str) -> HoldemGame:
    return HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS),
                      encoder=make_encoder(encoder))


def static_part(hands: int, seed: int, equity_samples: int) -> dict:
    game = HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS))
    encs = {name: make_encoder(name) for name in CANDIDATES}
    out = {"corpus": {"hands": hands, "seed": seed, "bet_menu": game.signature(),
                      "policy": "random legal: fold 0.15 / check-call 0.5 / bets share the rest"}}
    t = time.perf_counter()
    corpus = generate_corpus(game, hands, seed)
    comp = measure_compression(game, corpus, encs)
    facts = state_facts(game, corpus, equity_samples)
    out["compression"] = comp.to_dict()
    out["quality"] = {n: measure_quality(game, corpus, facts, e) for n, e in encs.items()}
    out["quality"]["raw"] = measure_quality(game, corpus, facts, RawHoldemEncoder())
    out["recall_audit"] = {}
    for n, e in list(encs.items()) + [("raw", RawHoldemEncoder())]:
        a = audit_perfect_recall(game, corpus, e, n)
        d = a.to_dict()
        d["perfect_recall_by_design"] = getattr(e, "perfect_recall_by_design", n in ("raw", "bucket"))
        out["recall_audit"][n] = d
    lines = {}
    for name, line in FIXED_LINES.items():
        lc = generate_line_corpus(game, line, hands, seed + 1)
        lfacts = state_facts(game, lc, equity_samples)
        lcomp = measure_compression(game, lc, encs)
        lines[name] = {"line": "/".join(line), "states": lcomp.states,
                       "raw_keys": lcomp.raw_keys, "keys": lcomp.keys,
                       "compression": {n: lcomp.raw_keys / k for n, k in lcomp.keys.items()},
                       "quality": {n: measure_quality(game, lc, lfacts, e)
                                   for n, e in encs.items()}}
        print(f"[A] {name}: {lcomp.keys}", flush=True)
    out["fixed_line_corpora"] = lines
    out["seconds"] = time.perf_counter() - t
    return out


def runs_of(encoder: str, runs_dir: Path, bucket_dir: Path):
    if encoder == "bucket":
        files = sorted(bucket_dir.glob("run_seed*.jsonl"))
    else:
        files = sorted(runs_dir.glob(f"{encoder}_seed*.jsonl"))
    runs = {}
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rows:
            runs[rows[0]["seed"]] = rows
    return runs


def training_part(runs_dir: Path, bucket_dir: Path, top_n: int = 2000) -> dict:
    spots = canonical_spots()
    out = {}
    for enc in CANDIDATES:
        runs = runs_of(enc, runs_dir, bucket_dir)
        if not runs:
            continue
        game = game_for(enc)
        res = {"runs": {}, "seed_disagreement": {}}
        at = {}
        for seed, rows in sorted(runs.items()):
            per, prev_n, prev_pols, prev_it = [], 0, None, 0
            for r in rows:
                solver = load_checkpoint(Path(r["checkpoint"]), game)
                v = visit_counts(solver)
                pols = spot_policies(game, solver.infosets, spots)
                row = {
                    "iterations": r["iterations"],
                    "wall_clock_seconds_cumulative": r["total_seconds"],
                    "iterations_per_second_segment": r["iters_per_sec_segment"],
                    "new_infosets_since_previous": r["infosets"] - prev_n,
                    "new_infosets_per_iteration": (r["infosets"] - prev_n) / (r["iterations"] - prev_it),
                    "checkpoint_bytes": Path(r["checkpoint"]).stat().st_size,
                    "infoset_memory_estimate_bytes": int(r["memory_mb_estimate"] * 1e6),
                    "entropy_bits_visit_weighted": r["entropy_bits_visit_weighted"],
                    "top_n_prev_mean_l1": r.get("top_n_prev_mean_l1"),
                    "visits": visit_stats(v, prev_n),
                    "canonical": {},
                }
                for i, p in enumerate(pols):
                    rec = policy_record(p)
                    if prev_pols is not None and p.visited and prev_pols[i].visited:
                        rec["l1_vs_previous_checkpoint"] = round(l1(prev_pols[i], p), 4)
                    row["canonical"][p.spot] = rec
                row["canonical_visited"] = sum(1 for p in pols if p.visited)
                per.append(row)
                if r["iterations"] in (1000, 5000):
                    at.setdefault(r["iterations"], {})[seed] = (top_visited(solver, top_n), pols)
                prev_n, prev_pols, prev_it = r["infosets"], pols, r["iterations"]
                print(f"[C] {enc} seed {seed} it {r['iterations']}: "
                      f"{row['visits']['counts']} canon {row['canonical_visited']}", flush=True)
                del solver
            res["runs"][str(seed)] = per
        for it, by_seed in sorted(at.items()):
            if len(by_seed) < 2:
                continue
            seeds = sorted(by_seed)
            pairs, spot_rows = {}, {}
            for i, sp in enumerate(spots):
                vis = [s for s in seeds if by_seed[s][1][i].visited]
                spot_rows[sp.name] = {
                    "visited_by": vis,
                    "category": "all" if len(vis) == len(seeds) else ("none" if not vis else "some"),
                    "pairwise_l1": {f"{a}v{b}": (round(l1(by_seed[a][1][i], by_seed[b][1][i]), 4)
                                                 if by_seed[a][1][i].visited and by_seed[b][1][i].visited
                                                 else None)
                                    for a, b in itertools.combinations(seeds, 2)}}
            for a, b in itertools.combinations(seeds, 2):
                ta, tb = by_seed[a][0], by_seed[b][0]
                common = sorted(set(ta) & set(tb))
                d = [float(np.abs(ta[k][1] - tb[k][1]).sum()) for k in common]
                sd = [e["pairwise_l1"][f"{a}v{b}"] for e in spot_rows.values()
                      if e["pairwise_l1"][f"{a}v{b}"] is not None]
                pairs[f"seed {a} vs seed {b}"] = {
                    "canonical_spots_visited_by_both": len(sd),
                    "canonical_mean_l1": float(np.mean(sd)) if sd else None,
                    f"top{top_n}_overlap_fraction": len(common) / top_n,
                    f"top{top_n}_common_mean_l1": float(np.mean(d)) if d else None}
            res["seed_disagreement"][str(it)] = {
                "seeds": seeds, "max_possible_l1": 2.0, "pairs": pairs,
                "canonical_spots": spot_rows,
                "spot_counts": {c: sum(1 for e in spot_rows.values() if e["category"] == c)
                                for c in ("all", "some", "none")}}
        out[enc] = res
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--bucket-runs-dir", type=Path, required=True)
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equity-samples", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "validation" / "abstraction_v2.json")
    args = p.parse_args()
    doc = {
        "format": "pokeralpha.abstraction_v2/v1",
        "encoders": {n: make_encoder(n).signature() for n in CANDIDATES},
        "static": static_part(args.hands, args.seed, args.equity_samples),
        "training": training_part(args.runs_dir, args.bucket_runs_dir),
        "notes": {
            "bucket_training": "bucket rows reuse the Phase 25 runs (same game, seeds, budgets); "
                               "they predate the platform-independent strategy_dot change",
            "equity_reference": "within-key equity is vs a uniformly random hand, offline only",
            "exploitability": "not computed",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
