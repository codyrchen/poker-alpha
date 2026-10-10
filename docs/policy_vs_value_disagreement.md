# Policy disagreement vs value disagreement (final-trust project, Phase 8-10)

Question: does the large strategy-L1 disagreement between seeds correspond
to strategically meaningful EV differences? Data:
`results/validation/policy_vs_value_disagreement_{native_1m,release_200k}.json`,
`preflop169_value_native_1m.json` (84 canonical states × 5,000 duplicate
deals per action; 169-class BTN study × 2,000). EV semantics: self-play
value of the artifact family in the real game — a decision-consequence
proxy, not exploitability.

## Headline (1M native, canonical suite)

* 57 / 84 states have high policy L1 (≥ 0.3). **Only 7 carry statistically
  significant EV consequence** (cross-seed regret ≥ max(0.5% pot, 3·SE)).
  50 / 57 are "high L1 / low EV consequence": seeds mix differently between
  near-equal actions.
* Spearman(policy L1, cross-regret) = **0.49** on the 169-class BTN study —
  policy L1 is a weak proxy for decision consequence.
* Median action margin (best − second EV): 0.30 bb at BTN first action —
  most preflop decisions are nearly value-flat across the abstract menu,
  which is exactly where MCCFR averages mix arbitrarily.
* Across milestones the consequences shrink: significant-consequence states
  24 (200k) → 8 (1M); their summed cross-regret 102 → 57 bb; median
  cross-regret 1.91 → 1.09 bb.

## The consequential cluster (1M)

| state | policy L1 | max cross-regret (bb) | note |
|---|---|---|---|
| BTN vs jam :: A5s | 0.33 | 14.8 | call-EV depends strongly on which seed's jam range you face |
| BB vs jam :: 22 | 0.48 | 6.4 | same mechanism |
| BB vs x350 :: QQ | 0.94 | 3.9 | 3-bet-defense mix |
| BB vs x350 :: JJ | 0.85 | 3.4 | |
| BTN unopened :: 93s | 1.12 | 3.4 | open-or-fold junk boundary |
| BTN unopened :: A5s | 1.49 | 3.1 | |
| BB vs x350 :: KK | 1.21 | 2.7 | |

Mechanism: in jam/3-bet-defense spots the *value of calling* depends on the
opponent's jam/3-bet range, which still differs between seeds — so policy
uncertainty there is genuine EV uncertainty. Everywhere else the mixing is
between actions whose EVs differ by less than the noise floor.

## Exact-game cross-check (Phase 12)

The same signature appears in exact, perfect-recall, abstraction-free
reduced games (`confidence_calibration_dataset.json`, 30k iterations,
5 seeds): 37% of preflop infosets have seed disagreement ≥ 0.5, half of
them with EV regret < 2% of pot; disagreement concentrates where the exact
action margin is small (median margin 0.23 among high-disagreement vs 0.41
overall). **High-L1 mixing is intrinsic to finite-sample external-sampling
MCCFR whenever actions are near-equivalent — no abstraction, imperfect
recall, or implementation issue is needed to produce it.** Regret among
those states still shrinks with training (0.10 → 0.04 bb from 3k → 30k).

## Consequence for confidence methodology

Confidence should weight *decision consequence*, not mixing aesthetics:

* seed disagreement remains useful — it is the best wrong-action predictor
  (rho 0.39) and best L1 predictor (0.57);
* visit count is the strongest EV-regret predictor (rho −0.35, AUC 0.74) —
  reinstated as a real signal, not just a floor;
* checkpoint movement predicts EV regret at **no horizon** (|rho| ≤ 0.05);
  its only defensible role is a small conjunction filter (see
  `docs/solver_confidence_v2.md`);
* action-value margin is environment information: small margins mean
  disagreement is likely harmless, large margins (jam spots) mean it
  matters. Stored for reporting where measured; not a per-key table signal
  (computing q for 140k keys is not tractable).
