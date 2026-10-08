"""Phase 40: assemble results/validation/release_candidate.json.

Collects the release configuration, strategy artifact provenance (command,
checksum, config signature, iterations, seed), and per-area statuses and
readiness from the committed result files. Test counts and CI status are
passed in (they come from running the suite and from GitHub).

Usage::

    python experiments/release_candidate.py --primary v2 --tests-nonslow "N passed, K skipped" \\
        --tests-slow "M passed" --ci "green on <sha>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results" / "validation"


def load(name):
    p = V / name
    return json.loads(p.read_text()) if p.exists() else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", choices=("v1", "v2"), required=True)
    p.add_argument("--tests-nonslow", required=True)
    p.add_argument("--tests-slow", required=True)
    p.add_argument("--ci", required=True)
    a = p.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from poker_alpha.solver_config import PRIMARY_CONFIG, V2_CONFIG
    from poker_alpha.solvers.strategy_artifact import load_artifact

    cfg = V2_CONFIG if a.primary == "v2" else PRIMARY_CONFIG
    art_path = ROOT / "results" / "strategy" / ("holdem_v2_seed0.npz" if a.primary == "v2" else "holdem_v1_seed0.npz")
    art = load_artifact(art_path, cfg.build_game())
    conf_path = art_path.with_name(art_path.stem + "_confidence.npz")
    cmd = ("python experiments/holdem_mccfr_validation.py --v2-config --seed 0 --milestones "
           "1000,5000,10000,30000,100000 --ckpt-dir DIR --out DIR/v2_seed0.jsonl; then "
           "python experiments/phase29_analysis.py --config v2 --runs-dir DIR --artifact OUT.npz") \
        if a.primary == "v2" else \
        ("python experiments/holdem_mccfr_validation.py --locked-config --seed 0 --milestones "
         "1000,3000,10000,30000,100000 ...; --resume ... --milestones 300000; then phase29_analysis.py")
    train = load(f"holdem_training_{'v2' if a.primary == 'v2' else 'v1_300k'}.json")
    gate = load("solver_gate_calibration.json")
    doc = {
        "format": "pokeralpha.release_candidate/v1",
        "solver": {
            "config_signature": cfg.signature(), "config": cfg.to_dict(),
            "recall": "IMPERFECT RECALL - no standard CFR equilibrium guarantee",
            "averaging": cfg.averaging, "sampling": cfg.sampling,
            "artifact": {"path": str(art_path.relative_to(ROOT)), "sha256": sha256(art_path),
                         "bytes": art_path.stat().st_size, "iterations": art.meta.get("iterations"),
                         "seed": art.meta.get("seed"), "infosets": len(art),
                         "generation_command": cmd,
                         "validation_status": "EXPERIMENTAL (abstract strategy; exploitability of the "
                                              "full abstraction not computed; gated at lookup time)"},
            "confidence_table": {"path": str(conf_path.relative_to(ROOT)), "sha256": sha256(conf_path)}
            if conf_path.exists() else None,
            "gate_thresholds": gate["thresholds"] if gate else None,
            "training_summary": {s: {"iterations": r[-1]["iterations"], "infosets": r[-1]["infosets"]}
                                 for s, r in (train or {}).get("convergence_proxies", {}).items()},
        },
        "results_files": sorted(f.name for f in V.glob("*.json")),
        "tests": {"non_slow": a.tests_nonslow, "slow": a.tests_slow},
        "ci": a.ci,
        "statuses": {
            "VALIDATED": [
                "Kuhn/Leduc research core (exact exploitability, pinned digests)",
                "Hold'em rules engine (2-9 seats, side pots, all-in) and solver-game rules/utility checks",
                "external-sampling MCCFR correctness on exact reduced Hold'em games (preflop + 6 river subgames)",
                "config-bound checkpoints and strategy artifacts (signature checks, config inference)",
                "rollout estimator vs closed forms (within 3 SE, error ~1/sqrt(n))",
                "unified pipeline, solver -> rollout -> heuristic cascade, solver-confidence gate",
                "legal-size filter: no recommendation below the NLHE minimum bet/raise",
            ],
            "PARTIALLY VALIDATED": [
                "compact abstraction: river error measured and reduced (v2), flop/turn error not measured exactly",
                "solver-confidence thresholds (calibrated on exact games, applied to the full abstraction)",
                "decision engine response models (rollouts disagree with exact river equilibria in 16/36 spots)",
                "screen observer on synthetic images",
            ],
            "EXPERIMENTAL": [
                "trained abstract HU strategy (imperfect recall, not converged, full exploitability unknown)",
                "range / opponent modelling (recommendations change with the assumed model)",
            ],
            "BLOCKED": ["real PokerNow screen recognition (0 real annotated fixtures)"],
        },
        "readiness": {
            "research": "READY WITH CAVEATS",
            "hand_analysis": "READY WITH MODEL CAVEATS",
            "private_play_money_test_decision_support": "READY WITH MANUAL STATE VERIFICATION",
            "real_screen_observation": "BLOCKED ON REAL FIXTURES",
            "trusted_solver_recommendations": "EXPERIMENTAL",
        },
        "not_claimed": ["solved NLHE", "GTO / Nash equilibrium of Hold'em", "profitable bot",
                        "real PokerNow accuracy", "exploitability of the full trained abstraction"],
    }
    (V / "release_candidate.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["solver"]["artifact"], indent=1))


if __name__ == "__main__":
    main()
