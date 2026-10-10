"""Compare v2 release candidates: seed 0 at 100k / 200k / 300k.

Reads strategy artifacts and confidence tables (paths below, override with
--dir) plus the committed Phase 65 analysis
(results/validation/holdem_training_v2_300k.json) and writes
results/validation/release_candidates_v2.json:

config / game / encoder signatures, strategy compatibility with the release
config, confidence-table compatibility (signature, construction, key
coverage), canonical policy movement, seed disagreement, gate acceptance,
artifact size and load time, cross-play.

    python experiments/compare_release_candidates.py --dir results/strategy
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from gate_acceptance import acceptance  # noqa: E402

from poker_alpha.decision.solver_gate import ConfidenceTable, GateThresholds  # noqa: E402
from poker_alpha.pipeline import load_solver  # noqa: E402
from poker_alpha.solver_config import RELEASE_CONFIG  # noqa: E402
from poker_alpha.solvers.strategy_artifact import load_artifact  # noqa: E402
from poker_alpha.utils.provenance import file_sha256  # noqa: E402

CANDIDATES = {
    "100k": ("holdem_v2_seed0.npz", "holdem_v2_seed0_confidence.npz"),
    "200k": ("holdem_v2_seed0_200k.npz", "holdem_v2_seed0_200k_confidence.npz"),
    "300k": ("holdem_v2_seed0_300k.npz", None),
}


def timed_load(path, confidence):
    times = []
    for _ in range(3):
        t = time.perf_counter()
        if confidence is None:
            from poker_alpha.decision import SolverStrategyProvider
            prov = SolverStrategyProvider.from_artifact(path, RELEASE_CONFIG, use_gate=False)
        else:
            prov = load_solver(path)
        times.append(time.perf_counter() - t)
    return prov, round(statistics.median(times), 3)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", type=Path, default=ROOT / "results" / "strategy")
    p.add_argument("--dir100k", type=Path, default=ROOT / "results" / "strategy")
    p.add_argument("--out", type=Path, default=ROOT / "results/validation/release_candidates_v2.json")
    a = p.parse_args(argv)
    game = RELEASE_CONFIG.build_game()
    analysis = json.loads((ROOT / "results/validation/holdem_training_v2_300k.json").read_text())
    th = GateThresholds.calibrated()
    rows = {}
    for name, (art_f, conf_f) in CANDIDATES.items():
        d = a.dir100k if name == "100k" else a.dir
        path = d / art_f
        art = load_artifact(path, game)              # raises on any signature mismatch
        row = {"artifact": art_f, "bytes": path.stat().st_size, "sha256": file_sha256(path),
               "format": art.format, "seed": art.meta.get("seed"),
               "iterations": art.meta.get("iterations"), "infosets": len(art.strategy),
               "config_signature": art.config_signature,
               "game_signature": art.game_signature, "encoder_signature": art.encoder_signature,
               "compatible_with_release_config": art.config_signature == RELEASE_CONFIG.signature()}
        if conf_f is not None:
            cpath = d / conf_f
            table = ConfidenceTable.load(cpath, art.config_signature)
            keys = set(art.strategy)
            covered = sum(1 for k in keys if k in table.stats)
            row["confidence_table"] = {
                "file": conf_f, "bytes": cpath.stat().st_size, "sha256": file_sha256(cpath),
                "config_signature_matches": table.config_signature == art.config_signature,
                "construction": table.meta,
                "built_for_this_iteration": table.meta.get("final") == art.meta.get("iterations"),
                "artifact_key_coverage": round(covered / len(keys), 4)}
            acc = acceptance(table, th)
            row["gate_acceptance_visit_weighted"] = {
                s: acc[s]["share_visit_weighted"] for s in ("preflop", "flop", "turn", "river", "all")}
            row["preflop_first_action_keys"] = acc["preflop_first_action_keys"]
        else:
            row["confidence_table"] = None
            row["gate_acceptance_visit_weighted"] = None
        prov, secs = timed_load(path, conf_f and (d / conf_f))
        row["load_seconds_median_of_3"] = secs
        row["load_includes"] = "artifact + checksums + confidence table" if conf_f else \
            "artifact + checksums (no confidence table exists)"
        rows[name] = row
    # movement / disagreement / cross-play from the committed analysis
    seed0 = {r["iterations"]: r for r in analysis["convergence_proxies"]["0"]}
    movement = {f"{a_}->{b_}": seed0[b_]["matrix_l1_vs_previous"]
                for a_, b_ in ((100000, 150000), (150000, 200000), (200000, 250000),
                               (250000, 300000))}
    dis = {it: {k: v["matrix_both_visited"]["mean_l1"] for k, v in
                analysis["seed_disagreement"][str(it)]["pairs"].items()}
           for it in (100000, 200000)}
    cross = [{k: c[k] for k in ("match", "a", "b", "bb_per_100", "ci95_bb_per_100")}
             for c in analysis["crossplay"]["matches"]]
    res = {"format": "pokeralpha.release_candidates/v1",
           "release_config": RELEASE_CONFIG.signature(), "candidates": rows,
           "canonical_policy_movement_seed0_matrix_l1": movement,
           "seed_disagreement_matrix_mean_l1": {str(k): v for k, v in dis.items()},
           "seed_disagreement_300k": "not measurable: seeds 1 and 2 were trained to 200k only",
           "crossplay": cross, "crossplay_deals_per_match": analysis["crossplay"]["deals_per_match"]}
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for k, r in rows.items():
        print(k, r["infosets"], r["bytes"], r["load_seconds_median_of_3"],
              r["gate_acceptance_visit_weighted"] and r["gate_acceptance_visit_weighted"]["all"])
    print("movement", movement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
