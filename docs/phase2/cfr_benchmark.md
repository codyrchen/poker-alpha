# CFR-family benchmark under three normalizations (Phase II, Program 3)

Algorithms: vanilla CFR, CFR+, Discounted CFR (alpha 1.5, beta 0, gamma 2;
Brown & Sandholm 2019), Linear CFR, external-sampling MCCFR and
outcome-sampling MCCFR (epsilon 0.6; Lanctot et al. 2009). DCFR, LCFR and
OS-MCCFR are new in `poker_alpha/research/cfr_variants.py`. DCFR with CFR+
parameters reproduces CFR+ exactly (`tests/test_cfr_variants.py`).

Games: Kuhn, Leduc, reduced preflop Hold'em (AKQJ x 2 suits, 8 bb), a river
subgame (6x6 combos) and the auction duel (5 x 5). Deterministic methods
have 1 run each. Sampling methods have **10 seeds**. Every run records exact
exploitability at geometric node-visit checkpoints. Summaries interpolate in
log-log space at **equal iterations, equal node visits and equal wall time**,
with the evaluation clock excluded. Artifact:
`results/phase2/cfr_family_benchmark.json`. Reproduce:
`python experiments/phase2_cfr_family_benchmark.py --seeds 10` (about 10 min
on 10 workers; timings are per process under that load).

## Final exploitability at the largest common budget (chips)

| game (budget) | CFR | CFR+ | DCFR | LCFR | ES-MCCFR (10 seeds) | OS-MCCFR (10 seeds) |
|---|---|---|---|---|---|---|
| Kuhn (3e5 nodes) | 2.1e-3 | 5.8e-5 | 8.3e-5 | **3.3e-5** | 5.1e-3 ± 1.7e-3 | 4.0e-2 ± 2.5e-2 |
| Leduc (2e7 nodes) | 9.3e-3 | 2.4e-4 | **1.7e-4** | 6.4e-3 | 2.0e-2 ± 2.7e-3 | 0.47 ± 0.07 |
| Reduced preflop (1e7) | 2.1e-3 | 1.3e-4 | **5.1e-5** | 3.7e-4 | 4.2e-3 ± 5e-4 | 3.6e-2 ± 4.8e-3 |
| River subgame (2e6) | 3.6e-2 | 1.9e-3 | **7.0e-4** | 1.8e-3 | 4.6e-2 ± 1.4e-2 | 0.65 ± 0.23 |
| Auction (2e6) | 1.4e-4 | 5.8e-6 | **6.3e-9** | 1.7e-6 | 6.1e-4 ± 2.3e-4 | 2.9e-3 ± 8e-4 |

± is the standard deviation across seeds.

## Findings

1. **The normalization decides the ranking.** At equal *iterations*,
   ES-MCCFR looks 100-1,000x worse than full-traversal methods: Leduc
   0.735 vs 0.0002. One sampled iteration costs 13-43 node visits against
   9,451 (CFR) or 18,902 (CFR+, two traversals). At equal *nodes*, ES-MCCFR
   is within 1.3-4.4x of vanilla CFR on every game, and at mid budgets it is
   competitive with CFR+ (Leduc mid-grid: CFR 0.100, CFR+ 0.144,
   ES-MCCFR 0.171). Iteration-count comparisons across CFR variants are not
   meaningful.
2. **Discounted CFR is the best or tied-best large-budget method on 4 of 5
   games**, 1.4-2.7x better than CFR+ at equal nodes on the poker games, and by orders of
   magnitude on the auction duel. On Kuhn, LCFR and CFR+ are marginally
   better. Linear CFR is weak on Leduc (6.4e-3 vs 1.7e-4 for DCFR).
3. **At small node budgets, vanilla CFR beats CFR+.** This holds at the
   mid-grid point of Leduc and reduced preflop. CFR+'s two traversals per
   iteration and its linear averaging need enough iterations to pay off.
4. **Outcome-sampling MCCFR is the worst method everywhere** and has the
   highest seed variance (river: 0.65 ± 0.23). It is not a sensible choice
   in games small enough to traverse.
5. Memory is identical across methods: the same infoset tables, two
   float64 vectors per infoset.

## Re-measurement of the Phase I crossover claim

The Phase I claim (`RESEARCH.md` §3.3) is that CFR beats CFR+ below about
0.5 s of wall time on Leduc and CFR+ wins from about 1 s. Re-measured from
the recorded curves on this machine:

| budget | 0.05 s | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 5.0 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| CFR | 1.74 | 0.95 | 0.41 | 0.22 | **0.14** | 0.078 | 0.048 | 0.031 | 0.021 |
| CFR+ | n/a | 2.13 | 1.49 | 0.76 | 0.23 | **0.064** | 0.017 | 0.0065 | **0.0019** |

**The crossover reproduces qualitatively.** CFR is better up to 1 s and
CFR+ wins from 2 s, ending 11x better at 20 s. The crossover point is later
than Phase I's "about 1 s", most likely because of different hardware and
10 concurrent processes. The public claim's wording ("roughly 1 s") is
slightly optimistic for this setup. It should read "between 1 and 2 s on
the Phase II machine" if re-stated.
