# Autonomous weakness → experiment loop (Phase II, Program 11)

`poker_alpha/research/loop.py` and `experiments/phase2_research_loop.py`.
The loop runs MEASURE → FLAG → CLASSIFY (a transparent rule table, not
free-form judgement) → MANIFEST → RUN → COMPARE → RECORD. Every manifest
carries `"release_action": "none (human gate)"`. Artifacts:
`results/phase2/loop/plan.json`, `manifest_*.json`, `results.json`.
Registry entries are `loop_W1_*` and `loop_W2_*` in
`results/experiments/index.json`.

## Measured weaknesses and diagnoses

| weakness (confirmed evidence) | rule fired | likely source | experiment |
|---|---|---|---|
| W1 local sensitivity, deep-stack turn/river facing 0.9-2x pot (25.6x lift, confirmed) | sensitivity is associated with low visits (AUC 0.75) | training coverage | extend training 2M → 4M, re-measure |
| W2 seed disagreement on well-visited keys | *none* (0.197 < rule threshold 0.3) | unclassified | run anyway as a researcher-directed follow-up of Program 18 |
| W3 opponent-model sensitivity (41 %) | sensitivity only on the fallback path | rollout depends on the opponent model by design | none |
| W4 collision-dispersion signal | signal coverage below 5 % | confidence-system design | none |
| W5 off-tree sizes | translation failure rate at least 0.5 | action abstraction | action-menu study, later run as H4 |

## The full loop on W1: diagnosis falsified

Before the 4M artifacts existed, the size-boundary analysis (H3) found
that 80 % of the failing states change abstract key under the +10 %
perturbation, against 4.9 % of controls. A prediction was recorded
(`hypotheses.json`): W1 would come out NULL.

Experiment: three seeds of the release lineage were resumed bit-exactly
from their 2M checkpoints to 3M and 4M (about 13 min per million iterations
per seed under load). The measure is the solver-policy jump: TV between the
policies of the state's key and the perturbed state's key. It covers the 81
confirmed failing states and 328 controls from the same clusters.

| seed | failing jump 2M / 3M / 4M | control jump 2M / 4M | failing keys' median visits 2M → 4M |
|---|---|---|---|
| 0 | 0.345 / 0.338 / 0.332 | 0.029 / 0.022 | 5,171 → 11,499 |
| 1 | 0.381 / 0.347 / 0.325 | 0.031 / 0.025 | 6,185 → 9,605 |
| 2 | 0.343 / 0.340 / 0.335 | 0.025 / 0.022 | 6,252 → 12,899 |

**Classification: NULL, as predicted.** Doubling training barely moves the
failing jumps, which stay 12-15x the controls'. The failing keys are not
low-visit at the solver level, so the "training coverage" diagnosis was
wrong. The confirmed cause is the abstraction boundary (H3), and the useful
lever is translation (H4), not iterations. The loop worked as intended: it
proposed a plausible diagnosis, ran the controlled experiment and
falsified that diagnosis.

## W2: seed disagreement 2M → 4M

| key visits (4M) | keys | seed TV 2M | seed TV 4M | ratio |
|---|---|---|---|---|
| < 100 | 25,400 | 0.339 | 0.340 | 1.00 |
| 100-1k | 35,675 | 0.308 | 0.288 | 0.94 |
| 1k-10k | 46,299 | 0.241 | 0.218 | 0.90 |
| ≥ 10k | 30,444 | 0.165 | 0.145 | 0.88 |

The ratios match the 1M → 2M ratios (0.88-1.06) and are far above the 0.71
expected for sampling noise. Doubling training twice in a row removed
about 10 % of the disagreement each time. **Structural, confirmed on a
second doubling.**

The 3M/4M artifacts are in `results/strategy/candidates/phase2_extension/`.
They are research artifacts, **not release candidates**: no confidence
tables were built and no validation was run.
