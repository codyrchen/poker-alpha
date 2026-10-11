# Policy geometry and release-strategy forensics (Phase II, Programs 18, 51, 52, 57, 59)

Data: the release strategy and its sibling seeds, native MCCFR, seeds 0-2 at
1M and 2M iterations (`results/strategy/candidates/`). 144,025 stored
infosets; 143,105 present in all three 2M seeds. Artifact:
`results/phase2/policy_geometry.json`. Reproduce:
`python experiments/phase2_policy_geometry.py` (~6 s).

## Headline: release seed disagreement is mostly not training-limited

With uniform averaging, the 2M average is half the 1M average plus half the
1M-2M window. If seed-to-seed disagreement were sampling noise, the mean
pairwise TV between seeds would shrink by about 1/sqrt(2), a ratio of 0.71,
from 1M to 2M. Measured ratios (2M / 1M mean seed TV):

| keys by 2M visits | keys | seed TV at 1M | at 2M | ratio |
|---|---|---|---|---|
| < 20 | 16,194 | 0.315 | 0.335 | 1.06 |
| 20-100 | 19,898 | 0.353 | 0.343 | 0.97 |
| 100-1k | 42,200 | 0.315 | 0.291 | 0.92 |
| 1k-10k | 45,193 | 0.244 | 0.219 | 0.90 |
| >= 10k | 17,864 | 0.162 | 0.142 | 0.88 (per-key median 0.75) |

By street the ratio is 0.93 (preflop), 0.96 (flop), 0.93 (turn), 0.94 (river).
Only the most-visited keys approach the noise expectation. For the bulk of
the table, doubling training removed 4-12 % of the disagreement.

Disagreement is strongly associated with policy entropy: Spearman 0.45 over
all keys and **0.70 among keys with >= 1,000 visits**. Well-trained keys
that disagree across seeds are the mixed ones. The data are consistent with
two structural causes, which this analysis cannot separate:
near-indifference between actions (several near-equal-value actions, so
different seeds settle on different mixtures of equal value) and
non-convergence of CFR under the imperfect-recall abstraction. Both predict
that more iterations will not resolve it.

Extrapolation, clearly labelled: if preflop disagreement keeps decaying as
`T^b` with `b = log2(0.93) = -0.10`, halving it would take roughly 700x
the 2M iterations. This is a two-point fit and not a forecast.

Implication for the confidence gate: seed disagreement on well-visited
keys mostly measures *how mixed the policy is*, not how unfinished training
is. Program 4 showed seed disagreement is still the best single
false-confidence filter on exact games. On the release, though, a high value
on a heavily visited key should be read as "mixed or indifferent here", not
as "train more".

## Other measurements

* **Visits are extremely heavy-tailed.** The median is 647 visits, the top 1 % of
  keys receive a large share of all visits, and the bottom 50 % receive 1.6 %.
  The Hill tail index of the top 5 % is 1.64, a power-law-like tail with
  infinite variance. Keys below 20 visits: 11 %. Below 100: 26 %.
* **Low-visit keys are more mixed.** Spearman(visits, normalized entropy) is
  -0.34. Mean normalized entropy falls monotonically from 0.72 in the lowest
  visit decile to 0.27 in the highest. Part of this is that regret matching
  starts uniform and rarely visited keys have not moved far from it.
* **Entropy by street** and pure-policy shares are in `by_street`.
* **3-action simplex (24,014 keys).** 38 % are at a vertex (one action >= 0.95), 30 % on an
  edge (one action < 0.02) and 32 % in the interior. Interior policies keep
  a mean minimum action probability of 0.14.

## Limitations

Three seeds is a small sample for a disagreement estimate. Visits come from
the seed 0 2M run. TV distance is a policy distance and says nothing about
value: two different mixtures can be equally good. Program 4 and the
exact-game calibration data are where policy error is tied to regret.
