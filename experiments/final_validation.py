"""Phase 31: assemble results/validation/final_platform_validation.json.

Collects the measured evidence from the committed result files into one
status table (VALIDATED / PARTIALLY VALIDATED / EXPERIMENTAL / BLOCKED).
Test counts and CI status are passed in, because they come from running the
suite and from GitHub, not from a result file.

Usage::

    python experiments/final_validation.py --tests-passed N --tests-skipped K \\
        --slow-included --ci "green on <sha>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results" / "validation"


def load(name):
    p = V / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tests-passed", type=int, required=True)
    p.add_argument("--tests-skipped", type=int, required=True)
    p.add_argument("--slow-included", action="store_true")
    p.add_argument("--ci", required=True)
    a = p.parse_args()
    sel = load("solver_abstraction_selection.json")
    bench = load("backend_benchmark.json")
    train = load("holdem_training_v1.json")
    ext = load("holdem_training_v1_300k.json")
    final = load("final_benchmark.json")
    obs = load("observer_fixture_validation.json")

    seeds = train["seeds"]
    sanity = {s: sum(c["result"] == "PASS" for c in train["sanity_checks"][str(s)]) for s in seeds}
    n_checks = len(train["sanity_checks"][str(seeds[0])])
    cross = {f'{c["a"]} vs {c["b"]}': {"bb_per_100": round(c["bb_per_100"], 1),
                                        "ci95": [round(x, 1) for x in c["ci95_bb_per_100"]]}
             for c in train["crossplay"]["matches"]}
    components = [
        ("Kuhn / Leduc CFR research core", "VALIDATED",
         "exact exploitability; canonical strategy digests pinned and unchanged through Phases 26-31"),
        ("Hold'em rules engine (2-9 seats, side pots)", "VALIDATED",
         "rules invariants, chip conservation, side/split pots under test; no invariant failure"),
        ("Hand evaluator / equity", "VALIDATED",
         "evaluator checked against all 2,598,960 five-card hands; seeded equity with SE"),
        ("Scalable abstraction (compact encoder)", "PARTIALLY VALIDATED",
         f"only candidate passing all selection criteria; >=5-visit share "
         f"{sel['scorecard']['compact']['fraction_trained_ge5']:.1%} vs bucket "
         f"{sel['scorecard']['bucket']['fraction_trained_ge5']:.2%} at 5k; IMPERFECT RECALL "
         f"({sel['scorecard']['compact']['recall_violations']} of "
         f"{sel['scorecard']['compact']['colliding_keys']} colliding keys violate recall)"),
        ("Locked solver config + config-bound checkpoints/artifacts", "VALIDATED",
         "signature covers encoder, feature tables, bets, stack, raise cap, sampling, reference range; "
         "mismatches rejected in tests; frozen regression corpus and canonical suite"),
        ("Optimized MCCFR backend", "VALIDATED",
         f"{bench['speedup_iterations_per_second']:.2f}x it/s, bit-identical exact digests; "
         "no compiled backend built"),
        ("Trained abstract HU strategy", "EXPERIMENTAL",
         f"{len(seeds)} seeds; sanity checks {sanity} of {n_checks}; seeds close in cross-play; "
         "more training beats less; no exploitability, imperfect recall, policies not settled"),
        ("Decision engine (rollout / heuristic)", "PARTIALLY VALIDATED",
         "closed-form decision invariants pass; EVs depend on heuristic response models and a "
         "check-down assumption"),
        ("Range / opponent modelling", "EXPERIMENTAL",
         "Bayesian beliefs under heuristic priors and models; no ground-truth range accuracy"),
        ("Unified pipeline + source priority + uncertainty breakdown", "VALIDATED",
         "all four input sources reach DecisionReport through one path in tests; rejection codes "
         "for config mismatch, incompatible file, out of abstraction, unvisited, insufficient visits"),
        ("Screen observer on synthetic images", "PARTIALLY VALIDATED",
         "synthetic fixtures only; read-only"),
        ("Screen observer on real PokerNow", "BLOCKED",
         f"BLOCKED ON REAL FIXTURES: {obs['annotations'] if obs else 0} annotated screenshots; "
         "nothing measured, nothing fabricated"),
    ]
    doc = {
        "format": "pokeralpha.final_platform_validation/v1",
        "statuses": ["VALIDATED", "PARTIALLY VALIDATED", "EXPERIMENTAL", "BLOCKED"],
        "components": [{"component": c, "status": s, "evidence": e} for c, s, e in components],
        "solver": {
            "primary_encoder": sel["selected"]["PRIMARY_SOLVER_ENCODER"],
            "recall": sel["selected"]["recall"],
            "config_signature": sel["selected"]["config_signature"],
            "trained_iterations": train["final_iterations"],
            "extended_iterations": ext["final_iterations"] if ext else None,
            "artifact": train["artifact"],
            "sanity_checks_passed": sanity,
            "crossplay_bb_per_100": cross,
            "extension_crossplay_bb_per_100": (
                {f'{c["a"]} vs {c["b"]}': {"bb_per_100": round(c["bb_per_100"], 1),
                                            "ci95": [round(x, 1) for x in c["ci95_bb_per_100"]]}
                 for c in ext["crossplay"]["matches"]} if ext else None),
            "exploitability": "not computed",
        },
        "performance": {
            "mccfr_iterations_per_second": bench["optimized"]["iterations_per_second"],
            "baseline_iterations_per_second": bench["baseline"]["iterations_per_second"],
            "online": final["online"] if final else None,
        },
        "observer_real_pokernow": "BLOCKED ON REAL FIXTURES",
        "tests": {"passed": a.tests_passed, "skipped": a.tests_skipped,
                  "slow_included": a.slow_included},
        "ci": a.ci,
        "not_claimed": ["solved NLHE", "GTO / Nash equilibrium of Hold'em",
                        "profitable bot", "real PokerNow recognition accuracy",
                        "exploitability of the trained strategy"],
    }
    (V / "final_platform_validation.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["components"], indent=1))


if __name__ == "__main__":
    main()
