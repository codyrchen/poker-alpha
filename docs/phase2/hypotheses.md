# Self-generated research hypotheses (Phase II)

Each hypothesis is grounded in a measured Phase II anomaly, cited in
brackets. Rank combines scientific value, feasibility, information gain and
runtime. **Implemented** items have artifacts in `results/phase2/`.

| # | hypothesis | grounded in | rank | status |
|---|---|---|---|---|
| H1 | Averaging independently seeded strategies lowers exploitability; but not as much as spending the same compute on one seed | release disagreement is structural [P18] | 3 | **implemented**: -20-26 % vs one seed; 41-53 % worse than one seed at 3x compute |
| H2 | The release algorithm (ES-MCCFR, uniform averaging) also degrades with training under imperfect recall; discounting does not fix it | CFR+ degraded on IR Leduc [P5C] | 2 | **implemented**: ES-MCCFR 0.49 to 0.82, DCFR 0.46 to 0.94 on forget-round-1 |
| H3 | Confirmed local-sensitivity failures are abstraction-boundary crossings, not undertrained policies | sensitivity concentrated near 0.9-2x pot [P10] | 1 | **implemented**: key changes in 80 % of failing vs 4.9 % of control states |
| H4 | Randomized pseudo-harmonic translation is much cheaper than deterministic nearest-log translation | H3 plus the release matcher's design | 1 | **implemented and confirmed**: about 8x lower IP cost |
| H5 | More training (2M to 4M) does not shrink the confirmed failing states' sensitivity faster than controls (pre-registered) | H3 | 2 | **implemented** (Program 11 loop W1), see `loop/results.json` |
| H6 | Seed disagreement 2M to 4M stays structural (ratio well above 0.71) | ratio 0.88-0.94 from 1M to 2M [P18] | 2 | **implemented** (loop W2) |
| H7 | A value-based collision signal (shared-action regret) predicts error far better than strength-ladder dispersion in the release | P5D rho 0.93 vs 0.50; release signal inactive [P10] | 4 | open: needs per-key action values in full Hold'em |
| H8 | The texture feature can be merged at no value cost | its TV is below the noise floor [P8] | 6 | **refuted as evidence**: TV is not value [P17]; needs an exact flop/turn test |
| H9 | Equal-node comparisons reverse the CFR/CFR+ ordering at small budgets in every game | P3 Leduc mid-grid | 7 | partly measured (Leduc, reduced preflop) |
| H10 | DCFR's advantage grows with game depth (2-street vs 1-street) | P3 auction vs Leduc | 8 | open |
| H11 | Outcome-sampling's variance can be cut by a baseline / control variate (VR-MCCFR) | P3: OS variance 0.65 ± 0.23 | 9 | open (literature: Schmid et al. 2019) |
| H12 | The safe gadget's gap-closing plateau (22-32 %) rises with max-margin resolving | P6 plateau by 100 ms | 5 | open |
| H13 | Resolving after off-tree bets beats translation | H4 + P6 | 4 | **implemented and confirmed**: safe nested resolving -8-22 % vs pseudo-harmonic; unsafe 15-30x worse |
| H14 | Error-guided refinement keeps beating uniform under multi-street imperfect recall | P1 single-street only | 3 | open (needs a fast multi-street exact solver) |
| H15 | `equity_spread` refinement can be made value-aware cheaply (opponent-range-weighted spread) | P1: equity_spread second to br_gain | 10 | open |
| H16 | Positive regret is a refinement anti-signal because it tracks resolved mass, not heterogeneity | P1 negative | 11 | explained in doc, not tested further |
| H17 | Prioritized root sampling fails because variance lives below the root (opponent actions, later chance) | P2 null incl. oracle | 9 | open: prioritize opponent-action sampling |
| H18 | Rare-state targeting improves exact error on rare deals specifically even when the average does not move | P2 null averages | 12 | open |
| H19 | Low-visit keys are more mixed partly because regret matching starts uniform (initialisation bias) | P18 entropy falls with visits | 13 | open: test with a non-uniform init |
| H20 | The gate's 1.9 % ACCEPT rate on random states is driven by structural disagreement, so a margin-aware gate could accept more without risk | P10 + P18 | 6 | open: margin is not observable in the full game |
| H21 | Seed disagreement's predictive power on exact games (P4) does not transfer to the release's structural disagreement | P4 vs P18 | 7 | open |
| H22 | uint16 artifacts give identical decisions on all canonical spots | P8 TV 3.7e-5 | 14 | partly: 0 top-action flips |
| H23 | Array-backed lazy artifact loading cuts the 157 MB load RSS by 10x | P8 | 8 | open (systems) |
| H24 | The river table construction is now cheap enough that evaluating the true percentile per hand is unnecessary | P7 profile after T3 | 15 | open |
| H25 | A finer bet menu reduces red-team translation failures more than more training | P5B + P10 | 5 | open (needs retraining) |
| H26 | The Bayes agent's misspecification loss is large for hybrid opponents | P15: well-specified only | 10 | open |
| H27 | Recency decay should be tuned to the measured switch frequency | P15 regime shift | 12 | open |
| H28 | The meta-game Nash over agents is fragile to sampling noise without CI-aware solving | P15 bayes artifact | 11 | measured once (Bonferroni) |
| H29 | Distilled trees fail where a best responder can concentrate, so error should be weighted by BR sensitivity, not visits | P17 | 9 | open |
| H30 | The equal-iteration bias explains earlier claims that MCCFR "trails at every budget" | P3 vs RESEARCH.md §3.2 | 8 | measured: at equal nodes ES-MCCFR is within 1.3-4.4x of CFR |

Top five implemented, by rank: H3, H4, H2, H1 and H5/H6 (the Program 11
loop). Results are in the research report.
