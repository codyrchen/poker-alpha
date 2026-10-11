"""Autonomous weakness -> experiment loop (Phase II, Program 11).

MEASURE -> FLAG WEAKNESS -> CLASSIFY LIKELY SOURCE -> GENERATE EXPERIMENT
CONFIG -> RUN CONTROLLED EXPERIMENT -> COMPARE -> RECORD.

Diagnosis is a transparent rule table over measured signals (no free-form
model judgement). The loop never modifies or promotes the release: every
manifest carries ``"release_action": "none (human gate)"``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
LOOP_DIR = ROOT / "results" / "phase2" / "loop"


@dataclass
class Weakness:
    id: str
    description: str
    evidence: Dict[str, object]
    signals: Dict[str, float]          # normalized inputs to the rule table


@dataclass
class Diagnosis:
    weakness: str
    rule: str
    likely_source: str
    experiment: Optional[str]
    rationale: str


# Rule table: (name, predicate over signals, likely source, experiment kind)
RULES = [
    ("high_disagreement_high_visits",
     lambda s: s.get("seed_disagreement", 0) >= 0.3 and s.get("visits_median", 0) >= 1000,
     "abstraction / near-indifference (structural)", "extend_training_measure_disagreement"),
    ("sensitivity_low_visits",
     lambda s: s.get("auc_low_visits", 0.5) >= 0.65,
     "training coverage", "extend_training_remeasure_states"),
    ("high_translation_distance",
     lambda s: s.get("translation_fail_rate", 0) >= 0.5,
     "action abstraction (off-tree sizes)", "action_menu_study"),
    ("model_sensitivity_on_fallback_path",
     lambda s: s.get("model_sens_solver_path", 1.0) == 0.0 and s.get("model_sens_fallback_path", 0) > 0.3,
     "pathway: rollout fallback depends on the opponent model by design", None),
    ("inactive_signal",
     lambda s: s.get("signal_coverage", 1.0) < 0.05,
     "confidence-system design (signal defined on too few keys)", None),
]


def classify(w: Weakness) -> List[Diagnosis]:
    out = []
    for name, pred, source, exp in RULES:
        try:
            hit = pred(w.signals)
        except Exception:  # noqa: BLE001
            hit = False
        if hit:
            out.append(Diagnosis(w.id, name, source, exp, f"rule {name} fired on signals {w.signals}"))
    if not out:
        out.append(Diagnosis(w.id, "no_rule", "unclassified", None, "no rule matched; left for human review"))
    return out


def measure() -> List[Weakness]:
    """Weaknesses from committed Phase II artifacts (confirmed results only)."""
    r = ROOT / "results" / "phase2"
    conf = json.loads((r / "redteam_confirmation.json").read_text())
    ov = json.loads((r / "redteam_abstraction_overlap.json").read_text())
    geo = json.loads((r / "policy_geometry.json").read_text())
    ws = []
    top_local = conf["cluster_confirmation"]["local_sensitivity"][0]
    ws.append(Weakness(
        "W1_local_sensitivity_deep_stack", "confirmed local-sensitivity cluster: " + top_local["cluster"],
        {"confirmation_rate": top_local["confirmation_rate"], "lift": top_local["lift"],
         "state_level_confirmation": conf["state_confirmation"]["local_sensitivity"]["confirmation_rate"]},
        {"auc_low_visits": ov["objectives"]["local_sensitivity"]["k_visits"]["auc"]}))
    ws.append(Weakness(
        "W2_seed_disagreement_well_visited", "seed disagreement on keys with >= 1,000 visits",
        {"mean_tv_visits_ge_1000": geo["disagreement"]["mean_visits_ge_1000"],
         "spearman_vs_entropy_visits_ge_1000": geo["disagreement"]["spearman_vs_entropy_visits_ge_1000"]},
        {"seed_disagreement": geo["disagreement"]["mean_visits_ge_1000"], "visits_median": 1000.0}))
    ws.append(Weakness(
        "W3_model_sensitivity", "opponent-model sensitivity (41% of states)",
        {"population_rate": conf["population_failure_rates_confirmation"]["model_sensitivity"]},
        {"model_sens_solver_path": 0.0, "model_sens_fallback_path": 0.80}))
    ws.append(Weakness(
        "W4_collision_signal", "collision-dispersion confidence signal",
        {"keys_with_signal": ov["collision_signal_coverage"]["release_keys_with_collision"],
         "release_keys": ov["collision_signal_coverage"]["release_keys"]},
        {"signal_coverage": ov["collision_signal_coverage"]["release_keys_with_collision"] /
                            ov["collision_signal_coverage"]["release_keys"]}))
    tdist = conf["cluster_confirmation"]["translation_distance"][0]
    ws.append(Weakness(
        "W5_translation", "off-tree bet sizes: " + tdist["cluster"],
        {"confirmation_rate": tdist["confirmation_rate"]},
        {"translation_fail_rate": tdist["confirmation_rate"]}))
    return ws


def manifest(d: Diagnosis, config: dict) -> dict:
    return {"format": "pokeralpha.phase2.loop_manifest/v1", "weakness": d.weakness, "rule": d.rule,
            "likely_source": d.likely_source, "experiment": d.experiment, "config": config,
            "release_action": "none (human gate)", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def plan() -> dict:
    ws = measure()
    diags = {w.id: [asdict(x) for x in classify(w)] for w in ws}
    return {"format": "pokeralpha.phase2.loop_plan/v1", "weaknesses": [asdict(w) for w in ws], "diagnoses": diags}
