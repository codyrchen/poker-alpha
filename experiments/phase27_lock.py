"""Phase 27: select and lock the solver abstraction.

1. Scorecard of every Phase-26 candidate from
   ``results/validation/abstraction_v2.json`` (no new training) with explicit
   pass/fail criteria -> ``results/validation/solver_abstraction_selection.json``.
2. Frozen regression corpus for the locked config: seeded random-policy
   decision states with their expected infoset keys and legal actions
   -> ``tests/fixtures/solver_v1/regression_corpus.json``.
3. Canonical-state suite: expected keys / legal actions of the canonical
   spots under the locked config -> ``tests/fixtures/solver_v1/canonical_suite.json``.

Usage: ``python experiments/phase27_lock.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.solver_config import (PRIMARY_CONFIG,  # noqa: E402
                                       PRIMARY_SOLVER_ENCODER,
                                       PRIMARY_SOLVER_ENCODER_RECALL)
from poker_alpha.solvers.holdem_analysis import spot_state  # noqa: E402
from poker_alpha.validation.abstraction_audit import generate_corpus  # noqa: E402
from poker_alpha.validation.canonical_spots import canonical_spots  # noqa: E402

V2 = ROOT / "results" / "validation" / "abstraction_v2.json"
OUT = ROOT / "results" / "validation" / "solver_abstraction_selection.json"
FIX = ROOT / "tests" / "fixtures" / "solver_v1"
CORPUS_HANDS, CORPUS_SEED = 120, 27

# Gate thresholds (fixed before reading the scorecard; see docs/validation.md).
CRITERIA = {
    "revisit_ge5_fraction_min": 0.25,      # >= 25% of infosets trained >= 5 visits
    "revisit_ge5_vs_bucket_min_ratio": 10.0,
    "newly_discovered_fraction_max": 0.5,  # last segment is not mostly discovery
    "canonical_visited_all_seeds_min": 12,  # of 14
    "legal_mixing_max": 0.0,
    "equity_range_gt_0.5_share_max": 0.01,
    "equity_range_gt_0.3_share_max": 0.10,
    "mean_within_key_equity_std_max": 0.05,
}


def scorecard(doc: dict) -> dict:
    st, tr = doc["static"], doc["training"]
    rows = {}
    bucket_ge5 = tr["bucket"]["runs"]["0"][-1]["visits"]["fraction_trained_ge5"]
    for enc in doc["encoders"]:
        last = tr[enc]["runs"]["0"][-1]
        v = last["visits"]
        q = st["quality"][enc]
        mix = q["share_of_states_in_keys_mixing"]
        sd = tr[enc]["seed_disagreement"]
        top = sd.get("5000") or sd.get("1000")
        pairs = list(top["pairs"].values())
        row = {
            "signature": doc["encoders"][enc],
            "iterations": last["iterations"],
            "infosets": v["infosets"],
            "new_infosets_per_iteration_last_segment": last["new_infosets_per_iteration"],
            "fraction_newly_discovered_last_segment": v["fraction_newly_discovered"],
            "fraction_trained_ge5": v["fraction_trained_ge5"],
            "fraction_trained_ge20": v["fraction_trained_ge20"],
            "visit_share_in_infosets_ge5": v["visit_share_in_infosets_ge5"],
            "median_visits": v["median"],
            "canonical_visited_seed0": last["canonical_visited"],
            "canonical_visited_all_seeds": top["spot_counts"]["all"],
            "seed_disagreement_at": 5000 if "5000" in sd else 1000,
            "seed_canonical_mean_l1": sum(p["canonical_mean_l1"] for p in pairs) / len(pairs),
            "seed_top2000_overlap": sum(p["top2000_overlap_fraction"] for p in pairs) / len(pairs),
            "corpus_keys": st["compression"]["keys"][enc],
            "recall_violations": st["recall_audit"][enc]["violations"],
            "colliding_keys": st["recall_audit"][enc]["colliding_keys"],
            "perfect_recall": st["recall_audit"][enc]["violations"] == 0,
            "mean_within_key_equity_std": q["mean_within_key_equity_std"],
            "share_equity_range_gt_0.3": mix["equity_range_gt_0.3"],
            "share_equity_range_gt_0.5": mix["equity_range_gt_0.5"],
            "share_category_mixing": mix["category"],
            "share_legal_mixing": mix["legal"],
            "checkpoint_bytes": last["checkpoint_bytes"],
            "wall_clock_seconds": last["wall_clock_seconds_cumulative"],
        }
        checks = {
            "revisitation": row["fraction_trained_ge5"] >= CRITERIA["revisit_ge5_fraction_min"]
            and row["fraction_trained_ge5"] >= CRITERIA["revisit_ge5_vs_bucket_min_ratio"] * bucket_ge5,
            "not_mostly_discovery": row["fraction_newly_discovered_last_segment"]
            <= CRITERIA["newly_discovered_fraction_max"],
            "canonical_coverage": row["canonical_visited_all_seeds"]
            >= CRITERIA["canonical_visited_all_seeds_min"],
            "no_legal_mixing": row["share_legal_mixing"] <= CRITERIA["legal_mixing_max"],
            "collision_quality": row["share_equity_range_gt_0.5"] <= CRITERIA["equity_range_gt_0.5_share_max"]
            and row["share_equity_range_gt_0.3"] <= CRITERIA["equity_range_gt_0.3_share_max"]
            and row["mean_within_key_equity_std"] <= CRITERIA["mean_within_key_equity_std_max"],
        }
        row["checks"] = checks
        row["passes_all"] = all(checks.values())
        rows[enc] = row
    return rows


def regression_corpus(game) -> dict:
    corpus = generate_corpus(game, CORPUS_HANDS, CORPUS_SEED)
    states = [{"holes": [list(h) for h in s.holes], "board": list(s.board),
               "streets": list(s.streets), "contrib": list(s.contrib),
               "key": game.infoset_key(s), "legal": game.legal_actions(s)}
              for s in corpus]
    return {"format": "pokeralpha.solver_regression_corpus/v1",
            "config_signature": PRIMARY_CONFIG.signature(),
            "encoder_signature": game.encoder_signature(),
            "game_signature": game.signature(),
            "generator": {"function": "generate_corpus", "hands": CORPUS_HANDS,
                          "seed": CORPUS_SEED},
            "states": states}


def canonical_suite(game) -> dict:
    rows = []
    for sp in canonical_spots():
        s = spot_state(game, sp.position, sp.hole, sp.board, sp.streets)
        rows.append({"name": sp.name, "group": sp.group, "position": sp.position,
                     "hole": list(sp.hole), "board": list(sp.board),
                     "streets": list(sp.streets), "key": game.infoset_key(s),
                     "legal": game.legal_actions(s)})
    return {"format": "pokeralpha.canonical_suite/v1",
            "config_signature": PRIMARY_CONFIG.signature(), "spots": rows}


def main() -> None:
    doc = json.loads(V2.read_text())
    rows = scorecard(doc)
    passing = [e for e, r in rows.items() if r["passes_all"]]
    selected = PRIMARY_SOLVER_ENCODER
    if selected not in passing:
        raise SystemExit(f"HARD STOP: locked encoder {selected!r} fails the gate; passing={passing}")
    game = PRIMARY_CONFIG.build_game()
    out = {
        "format": "pokeralpha.solver_abstraction_selection/v1",
        "source": str(V2.relative_to(ROOT)),
        "criteria": CRITERIA,
        "scorecard": rows,
        "passing": passing,
        "selected": {
            "PRIMARY_SOLVER_ENCODER": selected,
            "recall": PRIMARY_SOLVER_ENCODER_RECALL,
            "config": PRIMARY_CONFIG.to_dict(),
            "config_signature": PRIMARY_CONFIG.signature(),
        },
        "gate_27_to_28": {
            "primary_selected": True,
            "canonical_postflop_repeated_visits": all(
                v["visits"] >= 5 for k, v in
                doc["training"][selected]["runs"]["0"][-1]["canonical"].items()
                if not k.startswith("BTN open")),
            "training_not_mostly_discovery": rows[selected]["checks"]["not_mostly_discovery"],
            "collision_quality_defensible": rows[selected]["checks"]["collision_quality"],
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "regression_corpus.json").write_text(json.dumps(regression_corpus(game), indent=0))
    (FIX / "canonical_suite.json").write_text(json.dumps(canonical_suite(game), indent=1))
    for e, r in rows.items():
        print(f"{e:20s} pass={r['passes_all']} {r['checks']}")
    print("selected", selected, out["gate_27_to_28"])


if __name__ == "__main__":
    main()
