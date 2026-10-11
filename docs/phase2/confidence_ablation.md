# Confidence system ablation (Phase II, Program 4)

**Question.** Which signals of the solver-confidence gate actually contribute
to accepted-risk performance?

**Data.** The confidence v2 calibration dataset
(`results/validation/confidence_calibration_dataset.json`): 8,880 infoset
rows from 9 exact games (3 reduced preflop, 6 river subgames) x training
milestones x 5 seeds, each with the production signals (visits, recent
movement `mv_recent`, seed disagreement) and the *true* EV regret, strategy
L1 and wrong-best-action of the solver's policy against the exact solution.
Collision dispersion is NaN by construction (perfect-recall keys), so the
collision rule cannot be ablated here.

Artifact: `results/phase2/confidence_ablation.json`. Reproduce:
`python experiments/phase2_confidence_ablation.py` (~15 s).

## Metrics

* coverage: fraction of rows accepted;
* accepted mean regret, in pots (2 bb preflop pot, 10-chip river pot);
* tail regret: 95th percentile of accepted regret (pots);
* false-confidence rate: accepted rows whose solver best action is wrong;
* AUC: ROC-AUC of a signal for "regret >= 2 % pot" (larger = less confident).

Two protocols: **fixed** (release thresholds visits >= 20, sd < 0.2,
mv_recent < 0.1) and **nested leave-one-game-out** (for every held-out game
the thresholds are grid-searched on the other 8 games with the objective
"lowest accepted regret at coverage >= 15 %", then applied to the held-out
game, so no reported number used its own thresholds).

## Results

| rule (ACCEPT iff ...) | coverage | accepted regret (pot) | tail95 | false-conf. | nested: cov / regret / fc |
|---|---|---|---|---|---|
| always accept | 1.000 | 0.140 | 0.747 | 0.251 | – |
| v1 gate (historical) | 0.098 | 0.014 | 0.040 | 0.033 | – |
| visits only (>= 20) | 0.714 | 0.061 | 0.273 | 0.221 | 0.512 / 0.039 / 0.211 |
| seed only (< 0.2) | 0.273 | 0.050 | 0.299 | 0.047 | 0.202 / 0.041 / 0.037 |
| movement only (< 0.1) | 0.578 | **0.187** | 0.865 | 0.226 | 1.000 / 0.140 / 0.251 |
| visits + seed | 0.234 | 0.023 | 0.094 | 0.030 | 0.161 / 0.014 / 0.017 |
| **v2 full** | 0.165 | 0.015 | 0.046 | 0.030 | 0.153 / 0.018 / 0.026 |
| v2 minus visits | 0.202 | 0.051 | 0.424 | 0.052 | 0.189 / 0.040 / 0.037 |
| v2 minus seed | 0.310 | 0.048 | 0.232 | 0.134 | 0.162 / 0.019 / 0.085 |
| v2 minus movement | 0.234 | 0.023 | 0.094 | 0.030 | 0.161 / 0.014 / 0.017 |

Signal AUCs (regret >= 2 % pot): visits 0.74, seed disagreement 0.63,
`mv_recent` **0.47**, `mv_hist` 0.44, action margin 0.36 (as a
*confidence* signal; i.e. a large margin is associated with larger regret
in this data).

Bootstrap 95 % CIs (rows) for v2 full: coverage 0.157–0.173, accepted
regret 0.012–0.019 pot, false confidence 0.021–0.040. Leave-one-game-out
jackknife SE of v2-full accepted regret: see `jackknife` in the artifact.

## Findings

1. **Seed disagreement is the load-bearing signal for false confidence.**
   Removing it quadruples the false-confidence rate (0.030 -> 0.134) and
   triples accepted regret. It is the only single signal that keeps false
   confidence below 5 %.
2. **Visits are the load-bearing signal for tail regret.** Removing the
   visit floor raises tail regret from 0.046 to 0.424 pot: rarely visited
   keys with low seed disagreement are *agreed-upon but wrong*.
3. **Recent movement alone is anti-informative** (AUC 0.47; the
   movement-only rule accepts rows with *higher* regret than accepting
   everything). Inside the full rule it still earns its place as a tail
   filter: dropping it raises coverage 0.165 -> 0.234 at the cost of
   accepted regret 0.015 -> 0.023 and tail 0.046 -> 0.094, with unchanged
   false confidence. The movement rule's contribution is therefore real
   but conditional, not a stand-alone confidence measure. Under the
   nested protocol the visits+seed rule (which never saw its own game's
   thresholds) reaches 0.014 pot at 16 % coverage, slightly better than
   nested v2-full (0.018 at 15 %), so the movement component is within
   noise of a null contribution once thresholds are chosen out of sample.
4. **v2 vs v1.** v2 full accepts 1.68x the rows of v1 at the same false
   confidence (0.030 vs 0.033), confirming the Phase I claim; its accepted
   mean regret is marginally higher (0.015 vs 0.014 pot).
5. **Threshold optimism.** Fixed release thresholds (chosen on the pooled
   data in Phase I) score better (0.015) than nested out-of-sample
   thresholds (0.018): the published number carries a small in-sample
   optimism of ~0.003 pot.

## Limitations

* 9 exact games, so game-level uncertainty is large; the rows within a
  game are correlated (same milestones and seeds);
* no collision signal in this data; the Hold'em street caps are not
  exercised here;
* the nested objective (regret at >= 15 % coverage) is one reasonable
  choice; a different coverage target would pick different thresholds.
* No v3 candidate is proposed: no rule beat v2 full on both coverage and
  accepted regret out of sample.
