"""Build apps/web/src/data/lab/phase2.json from committed Phase II artifacts.

Only numbers that exist in results/phase2/*.json are copied; nothing is
recomputed here.

    python scripts/build_phase2_web_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_web_research_data import envelope  # noqa: E402

R = ROOT / "results" / "phase2"
OUT = ROOT / "apps" / "web" / "src" / "data" / "lab" / "phase2.json"


def j(name):
    return json.loads((R / name).read_text())


def main():
    conf = j("adaptive_abstraction_confirm.json")["summary"]
    disc = j("adaptive_abstraction.json")["summary"]
    arms = ["baseline", "random", "uniform_matched", "adaptive:equity_spread", "adaptive:br_gain", "oracle", "raw"]
    refinement = {"arms": arms,
                  "confirmation": {a: conf["arms"][a]["exploitability_mean"] for a in arms},
                  "discovery": {a: disc["arms"][a]["exploitability_mean"] for a in arms},
                  "paired_vs_uniform": conf["paired"]["adaptive:br_gain"]["vs_uniform_matched"],
                  "paired_vs_random": conf["paired"]["adaptive:br_gain"]["vs_random_mean"],
                  "instances": conf["instances"],
                  "negative_signal": conf["arms"]["adaptive:positive_regret"]["exploitability_mean"]}
    nat = j("native_optimization_trials.json")
    prof = j("native_profile.json")
    plain = ["deduplicated river percentile table (no 1,081-value sort)",
             "+ board counted once per table in the evaluator",
             "+ flat open-addressing infoset index"]
    native = {"trials": [{"id": t["id"], "change": plain[i] if i < len(plain) else t["change"], "baseline": t["baseline_median"], "new": t["new_median"],
                          "speedup": t["speedup_median"], "identical": t["strategy_digest_identical"]} for i, t in enumerate(nat["trials"])],
              "profile_inclusive_before": prof["before"]["inclusive_pct"],
              "profile_self_before": prof["before"]["self_time_pct"]}
    res = j("local_resolving.json")
    budgets = res["budgets_ms"]
    resolving = {"budgets_ms": budgets, "settings": {}}
    for key, s in res["summary"].items():
        resolving["settings"][key] = {
            "blueprint": s["blueprint"]["br_value_mean"], "equilibrium": s["equilibrium"]["br_value_mean"],
            "unsafe": [s[f"unsafe@{b}"]["br_value_mean"] for b in budgets],
            "safe": [s[f"safe@{b}"]["br_value_mean"] for b in budgets],
            "full": [s[f"full@{b}"]["br_value_mean"] for b in budgets],
            "safe_worse_count": [s[f"safe@{b}"]["worse_than_blueprint_count"] for b in budgets],
            "unsafe_worse_count": [s[f"unsafe@{b}"]["worse_than_blueprint_count"] for b in budgets],
            "n": s["blueprint"]["n"]}
    tr, trc = j("translation.json")["summary"], j("translation_confirm.json")["summary"]
    translation = {m: {seat: {"discovery": tr[m][seat]["translation_cost_vs_restricted_equilibrium"],
                              "confirmation": trc[m][seat]["translation_cost_vs_restricted_equilibrium"]}
                       for seat in ("us=0", "us=1")} for m in tr}
    ac = j("active_compute_confirm.json")["summary"]
    negatives = [
        {"title": "Active compute allocation", "result": "NULL",
         "detail": "Prioritized deal sampling (4 signals + an exact best-response oracle) never replicated a gain over "
                   "standard MCCFR on independent seeds; effects matched an information-free random reweighting."},
        {"title": "Confidence v3 (logistic score)", "result": "REJECTED",
         "detail": "Lower accepted regret on 5/9 held-out games but a higher false-confidence rate on 8/9; worse under "
                   "game-family shift. v2 stays."},
        {"title": "Train longer under imperfect recall", "result": "NEGATIVE",
         "detail": "With round-1 history forgotten (Leduc), exploitability in the real game ROSE from 0.48 to 0.94 with CFR+, "
                   "and from 0.49 to 0.82 with the release's own algorithm."},
        {"title": "Policy distance as quality", "result": "NEGATIVE",
         "detail": "A distilled Leduc strategy only 0.064 TV from equilibrium is 3,600x more exploitable."},
        {"title": "Positive regret as a refinement signal", "result": "NEGATIVE",
         "detail": "No better than random splitting; error-guided refinement needs within-cell heterogeneity, not solver regret."},
    ]
    payload = {"refinement": refinement, "native": native, "resolving": resolving, "translation": translation,
               "negatives": negatives, "active_compute_note": {"games": list(ac)}}
    OUT.write_text(json.dumps(envelope("phase2_research", payload, {}), separators=(",", ":"), allow_nan=False))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
