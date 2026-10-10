"""Phase 1 of the solver-trust project: unambiguous release/candidate inventory.

For every strategy candidate in the repository: path, SHA-256, seed,
iterations, backend, signatures, infosets, matching confidence table (and
its construction), generation commit, size, and a real load test through
the standard loader. Writes results/validation/final_release_inventory.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.decision.solver_gate import ConfidenceTable  # noqa: E402
from poker_alpha.solver_config import (RELEASE_CONFIDENCE, RELEASE_CONFIG,  # noqa: E402
                                       RELEASE_ITERATIONS, RELEASE_STRATEGY)
from poker_alpha.solvers.strategy_artifact import load_artifact  # noqa: E402
from poker_alpha.utils.provenance import file_sha256  # noqa: E402

CANDIDATES = [
    # (label, artifact, confidence, status)
    ("v1_300k_seed0", "results/strategy/holdem_v1_seed0.npz",
     "results/strategy/holdem_v1_seed0_confidence.npz",
     "historical v1 release (locked reference for Phases 26-31)"),
    ("v2_100k_seed0", "results/strategy/holdem_v2_seed0.npz",
     "results/strategy/holdem_v2_seed0_confidence.npz",
     "previous v2 release (archived in PREVIOUS_RELEASE_STRATEGIES)"),
    ("v2_200k_seed0", "results/strategy/holdem_v2_seed0_200k.npz",
     "results/strategy/holdem_v2_seed0_200k_confidence.npz",
     "CURRENT OFFICIAL RELEASE"),
    ("v2_200k_seed1", "results/strategy/candidates/v2_extension/holdem_v2_seed1_200k.npz",
     None, "research (seed-disagreement companion)"),
    ("v2_200k_seed2", "results/strategy/candidates/v2_extension/holdem_v2_seed2_200k.npz",
     None, "research (seed-disagreement companion)"),
    ("v2_300k_seed0_python", "results/strategy/candidates/v2_extension/holdem_v2_seed0_300k.npz",
     None, "research candidate (python lineage, no confidence table)"),
    ("native_300k_seed0", "results/strategy/candidates/native_300k/holdem_v2_native_seed0_300k.npz",
     "results/strategy/candidates/native_300k/holdem_v2_native_seed0_300k_confidence.npz",
     "research candidate (native lineage)"),
    ("native_300k_seed1", "results/strategy/candidates/native_300k/holdem_v2_native_seed1_300k.npz",
     None, "research companion"),
    ("native_300k_seed2", "results/strategy/candidates/native_300k/holdem_v2_native_seed2_300k.npz",
     None, "research companion"),
    ("native_1m_seed0", "results/strategy/candidates/native_1m/holdem_v2_native_seed0_1000k.npz",
     "results/strategy/candidates/native_1m/holdem_v2_native_seed0_1000k_confidence.npz",
     "RECOMMENDED CANDIDATE (not promoted; see release_status.md)"),
    ("native_1m_seed1", "results/strategy/candidates/native_1m/holdem_v2_native_seed1_1000k.npz",
     None, "research companion"),
    ("native_1m_seed2", "results/strategy/candidates/native_1m/holdem_v2_native_seed2_1000k.npz",
     None, "research companion"),
]


def main() -> None:
    from poker_alpha.solver_config import config_for_signature
    from poker_alpha.solvers.strategy_artifact import read_config_signature

    out = {"format": "pokeralpha.final_release_inventory/v1",
           "release_constants": {
               "RELEASE_STRATEGY": RELEASE_STRATEGY,
               "RELEASE_CONFIDENCE": RELEASE_CONFIDENCE,
               "RELEASE_ITERATIONS": RELEASE_ITERATIONS,
               "RELEASE_CONFIG": RELEASE_CONFIG.signature()},
           "candidates": {}}
    problems = []
    for label, art_path, conf_path, status in CANDIDATES:
        p = ROOT / art_path
        row: dict = {"artifact": art_path, "status": status}
        if not p.exists():
            row["exists"] = False
            problems.append(f"{label}: missing {art_path}")
            out["candidates"][label] = row
            continue
        row["exists"] = True
        row["bytes"] = p.stat().st_size
        row["sha256"] = file_sha256(p)
        sig = read_config_signature(p)
        cfg = config_for_signature(sig)
        try:
            art = load_artifact(p, cfg.build_game() if cfg else None)
            row["load_test"] = "PASS"
            row["infosets"] = len(art)
            row["iterations"] = art.meta.get("iterations")
            row["seed"] = art.meta.get("seed")
            row["backend"] = art.meta.get("backend", "python")
            row["format_version"] = art.format
            row["config_signature"] = art.config_signature
            row["game_signature"] = art.game_signature
            row["encoder_signature"] = art.encoder_signature
            row["generation_commit"] = art.commit
        except Exception as exc:  # noqa: BLE001
            row["load_test"] = f"FAIL: {type(exc).__name__}: {exc}"
            problems.append(f"{label}: {row['load_test']}")
        if conf_path:
            cp = ROOT / conf_path
            crow: dict = {"path": conf_path, "exists": cp.exists()}
            if cp.exists():
                crow["sha256"] = file_sha256(cp)
                crow["bytes"] = cp.stat().st_size
                try:
                    table = ConfidenceTable.load(cp, row.get("config_signature"))
                    crow["load_test"] = "PASS"
                    crow["keys"] = len(table.stats)
                    crow["meta"] = table.meta
                    crow["methodology"] = ("solver_confidence/v1: visits + movement"
                                           "(final vs earlier) + seed disagreement + "
                                           "collision dispersion")
                    if row.get("iterations") and table.meta.get("final") != row["iterations"]:
                        problems.append(f"{label}: confidence final "
                                        f"{table.meta.get('final')} != artifact "
                                        f"iterations {row['iterations']}")
                except Exception as exc:  # noqa: BLE001
                    crow["load_test"] = f"FAIL: {type(exc).__name__}: {exc}"
                    problems.append(f"{label} confidence: {crow['load_test']}")
            else:
                problems.append(f"{label}: missing confidence {conf_path}")
            row["confidence_table"] = crow
        else:
            row["confidence_table"] = None
        out["candidates"][label] = row

    # Release pointer identity checks.
    rel = out["candidates"]["v2_200k_seed0"]
    out["release_identity_ok"] = (
        rel.get("load_test") == "PASS"
        and rel["artifact"] == RELEASE_STRATEGY
        and rel.get("iterations") == RELEASE_ITERATIONS)
    out["problems"] = problems
    dest = ROOT / "results/validation/final_release_inventory.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("load_test", "iterations", "seed", "backend",
                                    "infosets", "sha256")}
                      for k, v in out["candidates"].items()}, indent=1))
    print("problems:", problems or "none")
    print("wrote", dest)


if __name__ == "__main__":
    main()
