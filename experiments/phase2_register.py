"""Record every Phase II run in results/experiments/index.json (Program 13)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poker_alpha.research.runner import record  # noqa: E402

R = "results/phase2/"
RUNS = [
    ("p1_adaptive_abstraction_discovery", "discovery", "POSITIVE", [R + "adaptive_abstraction.json"], "seeds 0-4 x 6 boards",
     "br_gain refinement -31% vs uniform_matched, -57% vs random", "python experiments/phase2_adaptive_abstraction.py"),
    ("p1_adaptive_abstraction_confirmation", "confirmation", "POSITIVE", [R + "adaptive_abstraction_confirm.json"], "seeds 5000-5004 x 9 boards",
     "-24% vs uniform_matched (39/45 wins), -56% vs random (45/45)", "python experiments/phase2_adaptive_abstraction.py --confirm"),
    ("p1_adaptive_abstraction_highcap", "confirmation", "POSITIVE", [R + "adaptive_abstraction_highcap.json"], "seeds 0-2 x 6 boards",
     "-31% vs uniform_matched at K=8, 16 splits", "python experiments/phase2_adaptive_abstraction.py --k0 8 --splits 16 --hands 64 --seeds 3"),
    ("p2_active_compute_discovery", "discovery", "NULL", [R + "active_compute.json"], "seeds 0-9", "no replicated gain", "python experiments/phase2_active_compute.py"),
    ("p2_active_compute_confirmation", "confirmation", "NULL", [R + "active_compute_confirm.json"], "seeds 100-119",
     "discovery positives did not replicate", "python experiments/phase2_active_compute.py --seed-base 100 --seeds 20"),
    ("p3_cfr_family_benchmark", "descriptive", "DESCRIPTIVE", [R + "cfr_family_benchmark.json"], "10 seeds (sampling methods)",
     "DCFR best at large budgets on 4/5 games; crossover between 1 and 2 s", "python experiments/phase2_cfr_family_benchmark.py"),
    ("p4_confidence_ablation", "validation", "DESCRIPTIVE", [R + "confidence_ablation.json"], "LOGO over 9 games",
     "seed disagreement controls false confidence; visits control tail", "python experiments/phase2_confidence_ablation.py"),
    ("p4_confidence_v3_candidate", "validation", "NEGATIVE", [R + "confidence_v3_candidate.json"], "LOGO + family shift",
     "logistic score: higher false confidence on 8/9 games; keep v2", "python experiments/phase2_confidence_v3_candidate.py"),
    ("p5_abstraction_science", "descriptive", "DESCRIPTIVE", [R + "abstraction_capacity.json", R + "action_abstraction.json",
                                                              R + "recall_ablation.json", R + "collision_dispersion.json"],
     "3 seeds x 9 boards", "IR Leduc degrades with training; value-based collision predicts error", "python experiments/phase2_abstraction_science.py"),
    ("p6_local_resolving", "validation", "POSITIVE", [R + "local_resolving.json"], "seeds 7000-7002 x 9 boards",
     "safe gadget never worse (>1e-4) at >=1s; closes 9-31% of gap; unsafe 2-3x worse", "python experiments/phase2_local_resolving.py"),
    ("p7_native_optimization", "validation", "POSITIVE", [R + "native_optimization_trials.json", R + "native_profile.json"], "A/B, 3-5 reps",
     "1.96x training throughput, identical digest", "see docs/phase2/native_systems.md"),
    ("p8_artifact_compression", "descriptive", "DESCRIPTIVE", [R + "artifact_compression.json"], None,
     "uint16: -27% size, max TV 3.7e-5", "python experiments/phase2_artifact_compression.py"),
    ("p9_exhaustive_correctness", "validation", "VALIDATION", [R + "exhaustive_correctness.json"], None,
     "14.75M states, 0 violations", "python experiments/phase2_exhaustive_correctness.py"),
    ("p10_redteam_discovery", "discovery", "DESCRIPTIVE", [R + "redteam_discovery.json"], "state seed 101", "20,000 states",
     "python experiments/phase2_redteam.py"),
    ("p10_redteam_confirmation", "confirmation", "VALIDATION", [R + "redteam_confirmation.json", R + "redteam_abstraction_overlap.json"],
     "state seed 202", "17/18 clusters confirmed", "python experiments/phase2_redteam.py && python experiments/phase2_redteam_overlap.py"),
    ("p14_general_games", "validation", "VALIDATION", [R + "general_games.json"], None, "cheap-talk oracle holds",
     "python experiments/phase2_general_games.py"),
    ("p15_agents", "descriptive", "DESCRIPTIVE", [R + "agents.json"], "5 match seeds", "3 cycles survive Bonferroni",
     "python experiments/phase2_agents.py"),
    ("p16_llm_bench_mocks", "validation", "VALIDATION", [R + "llm_bench_mocks.json"], None, "all mock failure modes detected", "pytest tests/test_llm_bench_v2.py"),
    ("p17_strategy_compiler", "descriptive", "DESCRIPTIVE", [R + "strategy_compiler.json"], None,
     "0.064 TV distilled Leduc strategy is 3,600x more exploitable", "python experiments/phase2_strategy_compiler.py"),
    ("p18_policy_geometry", "descriptive", "DESCRIPTIVE", [R + "policy_geometry.json"], "3 seeds x {1M,2M}",
     "seed disagreement shrinks 4-12% per doubling", "python experiments/phase2_policy_geometry.py"),
    ("p21_native_stress", "validation", "VALIDATION", [R + "native_correctness_stress.json"], None, "10k parity bit-identical",
     "python experiments/phase2_native_stress.py --parts checkpoint,parity"),
    ("p22_evaluator_stress", "validation", "VALIDATION", [R + "evaluator_stress.json"], None, "all 6- and 7-card hands match",
     "python experiments/phase2_evaluator_stress.py"),
    ("p23_native_memory", "descriptive", "DESCRIPTIVE", [R + "native_memory.json"], "seed 11", "table saturates ~146k infosets, 131.5 MB",
     "python experiments/phase2_native_stress.py --parts scale"),
    ("h1_h3_hypotheses", "confirmation", "DESCRIPTIVE", [R + "hypotheses.json"], "8 reps", "ensembling -20-26% vs single, worse than 3x compute",
     "python experiments/phase2_hypotheses.py"),
    ("h4_translation", "confirmation", "POSITIVE", [R + "translation.json", R + "translation_confirm.json"], "seeds 900-902 / 4242-4244",
     "pseudo-harmonic ~8x cheaper than nearest-log (IP, release-like menu)", "python experiments/phase2_translation.py"),
]

if __name__ == "__main__":
    for name, mode, cls, outs, seeds, head, cmd in RUNS:
        record(name, status="done", mode=mode, classification=cls, outputs=outs, seeds=seeds, headline=head, command=cmd)
    print(len(RUNS), "runs recorded")
