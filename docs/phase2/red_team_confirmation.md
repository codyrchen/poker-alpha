# Red team Phase II: discovery, then confirmation (Program 10)

Phase I scored 400 random legal states. Phase II scores 20,000 for
discovery (state seed 101), then confirms on an independent 10,000-state
sample (seed 202). It also re-scores the top states with a different
decision seed and 4x the equity and rollout samples. Six objectives are
kept separate: gate rejection, seed disagreement, low visits (< 20),
translation distance (>= 0.3 from a trained size), local sensitivity
(TV >= 0.5 under a +10 % bet or pot perturbation) and opponent-model
sensitivity (TV >= 0.5 across archetypes).

Reproduce (about 8 min on 10 workers):

```bash
python experiments/phase2_redteam.py --n 20000 --confirm-n 10000 --workers 10
python experiments/phase2_redteam_overlap.py
```

Artifacts: `results/phase2/redteam_discovery.json` (with
`redteam_discovery_records.json.gz`), `redteam_confirmation.json`,
`redteam_clusters.json` (unbiased rates from the confirmation sample) and
`redteam_abstraction_overlap.json`.

## Population rates (discovery / confirmation)

| objective | discovery (20k) | confirmation (10k) |
|---|---|---|
| gate rejected | 51.5 % | 50.3 % |
| seed disagreement >= 0.5 | 49.9 % | 49.0 % |
| low visits | 0.47 % | 0.57 % |
| translation distance >= 0.3 | 13.2 % | 13.8 % |
| local sensitivity | 0.41 % | 0.47 % |
| model sensitivity | 41.3 % | 40.8 % |

Gate outcomes on discovery states: 1.9 % ACCEPTED, 18.1 % LOW CONFIDENCE
(solver), 28.5 % LOW CONFIDENCE (interpolated abstraction), 51.5 %
REJECTED (rollout fallback). These are random self-play states, not a model
of human play.

## Cluster confirmation

Clusters are street x SPR bucket x faced-bet bucket. For each objective, the
three clusters with the highest Wilson lower bound in discovery were
re-measured on the independent sample. **17 of 18 were confirmed** above
the population rate. The failure was "Flop, SPR < 1, unopened" for gate
rejection: 0.86 in discovery became 0.71 with n = 21, CI includes the base
rate. Largest confirmed effects:

| objective | cluster | confirmation rate (95 % CI) | lift |
|---|---|---|---|
| local sensitivity | Turn, SPR >= 8, bet 0.9-2x pot | 12.0 % (6.7-20.8) | 25.6x |
| local sensitivity | River, SPR >= 8, bet 0.9-2x | 10.3 % (4.1-23.6) | 21.8x |
| low visits | Preflop, SPR >= 8, bet 0.9-2x | 4.8 % (3.0-7.6) | 8.4x |
| translation distance | Turn, SPR < 1, bet < 0.4x | 100 % (78-100) | 7.3x |
| gate rejected | Preflop, SPR >= 8, bet 0.4-0.9x | 89.8 % (88.4-91.1), n = 1,946 | 1.8x |

## State-level confirmation

The top 40 discovery states per stochastic objective were re-scored with a
new decision seed and 4x the samples:

* local sensitivity: 36 of 40 confirmed (90 %); mean score 0.95 fell to 0.89, the expected regression to the mean;
* model sensitivity: 39 of 40 confirmed (97.5 %); mean 1.00 fell to 0.96.

## Connection to abstraction and training signals (10G)

Keys were recomputed for 19,768 states and joined with the release
confidence table. AUC of each key-level signal for separating failing from
non-failing states:

| objective | low visits | seed disagreement | movement | collision |
|---|---|---|---|---|
| local sensitivity | **0.75** | 0.61 | 0.57 | 0.47 |
| model sensitivity | 0.67 | **0.91** | 0.65 | 0.50 |
| translation distance | 0.57 | 0.43 | 0.42 | 0.47 |

* **Opponent-model sensitivity is a pathway effect, not a solver
  weakness.** It occurs on 80 % of gate-rejected states, which use the
  Monte Carlo rollout fallback, and on 0 % of states served by the solver.
  The rollout depends on the assumed opponent model by design. Seed
  disagreement predicts it only because seed disagreement drives the
  rejection.
* **Local perturbation sensitivity concentrates on low-visit keys** (AUC
  0.75), and on deep-stack (SPR >= 8) turn and river spots facing roughly
  pot-sized bets.
* **The collision-dispersion signal carries no information here** (AUC
  0.47-0.50). It is defined for only 2,057 of 144,025 release keys (1.4 %)
  and is 0 for 95 % of those. In the release, the collision component of
  confidence v2 is effectively inactive. Flop and turn abstraction risk is
  carried only by the street caps.

## Multiple comparisons

Discovery ranked about 100 clusters per objective. Ranks are selected and
therefore optimistic. Every number in the cluster table above is measured
on the independent sample, which was not used for selection.

## Limitations

State generation is random legal self-play. "Failure" thresholds are
conventions, and the rates scale with them. Sensitivity is a policy
distance, not a value loss.
