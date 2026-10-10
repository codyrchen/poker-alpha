# Direct native exact-game validation (final-trust project, Phase 3)

Previously the native backend's exact-reduced-game validation was
*transitive* (native ≡ Python bitwise on a shared random tape; Python
validated on exact games). This study runs native MCCFR **directly** on the
exact games. Data: `results/validation/native_direct_exact_validation.json`.
Command: `python experiments/native_direct_exact_validation.py` (~23 min).

## Method

The tabular adapter (`poker_alpha/native/tabular.py`, `cpp/src/tabular.cpp`)
flattens a root-chance Python `Game` into a betting tree + weighted deal
list + per-terminal utility matrices, every payoff and infoset key produced
by the *Python game itself* — zero rule duplication. The native solver runs
the same external-sampling MCCFR update code (regret matching, cdf
sampling, exactly-rounded dot) as the Hold'em backend. Exploitability of
the resulting average strategies is computed with the existing exact
best-response evaluator.

Games: the Phase-33 reduced preflop game (AKQJT × 2 suits, 10 bb) and all
six fixed-board 52-card river subgames. Reference: exact CFR+
(exploitability ≤ 0.001 everywhere). Samplers: Python MCCFR and native
MCCFR, 3 seeds each.

## Results

Matched 10,000-iteration exploitability (median of 3 seeds):

| game | Python | native | ratio |
|---|---|---|---|
| reduced preflop | 0.064 | 0.057 | 0.90 |
| river dry_high | 0.327 | 0.340 | 1.04 |
| river paired | 0.495 | 0.446 | 0.90 |
| river four-straight | 0.245 | 0.219 | 0.89 |
| river four-flush | 0.238 | 0.270 | 1.14 |
| river monotone | 0.259 | 0.272 | 1.05 |
| river low-connected | 0.362 | 0.410 | 1.13 |

Median ratio **1.04** — the two backends are statistically
indistinguishable at matched budgets (independent RNG streams; the spread
matches the seed-to-seed spread within either backend).

Native convergence continues as expected with more iterations
(exploitability at 300k, 3 seeds): preflop 0.007–0.010; rivers 0.019–0.067
(pots 2–10 bb) — the ~1/√T slope of the Phase-33 Python study.

## Verdict

**PASS — release blocker cleared.** Native MCCFR convergence on exact games
is now established directly, in addition to the bitwise tape parity. The
adapter and `cf_action_values` (exact counterfactual action values) also
power the confidence-calibration dataset
(`results/validation/confidence_calibration_dataset.json`).
