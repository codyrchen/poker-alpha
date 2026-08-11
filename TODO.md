# PokerAlpha — Development Plan

Working order follows the incremental plan in the project brief: each phase is
implemented smallest-correct-first, tested, smoke-run, and committed before the
next begins.

## Completed

1. Repository setup — package layout, pyproject, results dirs, seeding utils.
2. Kuhn Poker — extensive-form game with chance nodes, information sets,
   zero-sum utilities; 9 game tests.
3. Vanilla CFR — regret matching, cumulative regret/strategy, average strategy;
   deterministic full-tree traversal.
4. Strategy evaluation — expected value, imperfect-information best response,
   exploitability; 8 solver tests.
5. Kuhn convergence experiment — `experiments/kuhn_convergence.py`
   (CSV + log-log figure; verified value −1/18, exploitability 6.3e-4 @ 100k).

6. CFR+ (`solvers/cfr_plus.py`) — regret clipping (once per infoset per
   iteration; per-visit clipping measurably degrades to O(1/√T)), alternating
   updates, linear averaging; 5 tests; `experiments/cfr_comparison.py`
   (2.1e-6 vs 6.3e-4 exploitability at 100k, ~1.75× runtime cost).

## Current

7. External-sampling MCCFR; three-way comparison (convergence, runtime, memory).

## Next

8. Leduc Poker; convergence experiments; Kuhn-vs-Leduc scaling plots.
9. Card engine (Card/Deck/Hand) + Hold'em hand evaluator + tests.
10. Monte Carlo equity engine with seeded reproducibility.

## Future

11. Abstracted heads-up NL Hold'em state engine (restricted bet sizes).
12. Equity-bucket card abstraction; document information loss.
13. Real-time subgame solving; compute-vs-quality experiment.
14. Opponent archetypes (Balanced, Nit, Calling Station, Maniac, Bluff Heavy,
    Passive) via configurable behavior multipliers.
15. Observable opponent statistics (VPIP, PFR, aggression, fold-to-bet, ...)
    with no hidden-information leakage.
16. Bayesian opponent model (Beta-Bernoulli posteriors with explicit
    uncertainty); range estimation via Bayes rule over actions.
17. EV-derived exploitative strategy.
18. Risk-constrained adaptive strategy (confidence-weighted λ blend;
    exploitability guardrail).
19. Experiments: adaptation vs archetypes, opponent identification,
    distribution shift, exploration-vs-exploitation, overfitting-vs-sample-size,
    rake sensitivity.
20. Risk analytics: return metrics, Kelly, bankroll simulation, risk of ruin.
21. Performance profiling + optimization benchmark.
22. README finalization with benchmark tables from real runs.
23. RESEARCH.md writeup.
