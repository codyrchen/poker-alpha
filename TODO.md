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

7. External-sampling MCCFR (`solvers/mccfr.py`) — chance/opponent sampling,
   seeded determinism; 5 tests; three-way comparison experiment with both
   per-iteration and per-second views.

8. Leduc Poker (`games/leduc.py`) — public card, 2 rounds, raise cap,
   rank-merged infosets (288); 12 tests; converges to known value −0.0856.
   `experiments/leduc_convergence.py` measured: CFR 19.7 it/s vs MCCFR
   1,886 it/s (sampling advantage 2.3×→96× from Kuhn to Leduc); CFR+ 64×
   lower exploitability at 1k iters; MCCFR still trails exact CFR at matched
   wall-clock on a game this size (documented as an honest negative).

9. Card engine (`poker/cards.py`) — int-coded 52-card deck, Card/Deck/Hand,
   seeded shuffling, dead-card exclusion; 8 tests.
10. Hand evaluator (`poker/evaluator.py`) — all 9 categories as comparable
    tuples, wheel/steel-wheel, kickers, ties, best-of-5/6/7; 17 tests.
11. Monte Carlo equity engine (`poker/equity.py`) — uniform or weighted
    opponent ranges with blocker handling, seeded, std-error reporting;
    11 tests. Validated: AA 85.1% (known 85.2%), 72o 34.1% (known ~34.6%);
    ~7k sims/s (optimization target for the profiling phase).

12. Abstracted heads-up NL Hold'em (`games/holdem.py`) — 100 BB stacks,
    0.5/1 blinds, four streets, action set {f, c, b50, b100, b200, all-in},
    3-raise cap, pot-relative sizing, all-in runouts, sampled chance
    (`sample_chance`/`deal`; full chance enumeration intentionally
    unsupported). Betting bookkeeping replayed from the blinds forward as the
    single source of truth. 16 tests + 5,000-hand random-playout invariant
    check (chip conservation, stack caps, showdown equality).

13. Opponent-modeling core (`opponent/`), built on Leduc where every claim is
    exactly computable:
    - `archetypes.py` — OpponentConfig multiplier tilts of the equilibrium
      (balanced/nit/calling_station/maniac/bluff_heavy/passive).
    - `bayesian_model.py` — Beta-Bernoulli posteriors with calibrated
      uncertainty (mean/variance/credible intervals).
    - `beliefs.py` — discrete Bayes over candidate types from *public*
      actions, marginalizing hidden cards (showdowns reveal); posterior
      mixture strategy; entropy-based confidence.
    - `match.py` — seat-alternating match simulator with a hero-visible
      information firewall.
    - `exploit.py` — exact best response to the estimate, λ-blend with
      confidence×deviation weighting, exploitability-budget guardrail via
      bisection; `adaptive_strategy()`.
    Measured: BR vs maniac +1.08 chips/hand (equilibrium: +0.002) at 2000×
    the exploitability; guardrail holds blends within ε; maniac identified
    96% by 50 hands. 24 tests (115 total).

## Current

14. Equity-bucket card abstraction; document information loss.

## Future

13. Equity-bucket card abstraction; document information loss.
14. Real-time subgame solving; compute-vs-quality experiment.
15. Opponent archetypes (Balanced, Nit, Calling Station, Maniac, Bluff Heavy,
    Passive) via configurable behavior multipliers.
16. Observable opponent statistics (VPIP, PFR, aggression, fold-to-bet, ...)
    with no hidden-information leakage.
17. Bayesian opponent model (Beta-Bernoulli posteriors with explicit
    uncertainty); range estimation via Bayes rule over actions.
18. EV-derived exploitative strategy.
19. Risk-constrained adaptive strategy (confidence-weighted λ blend;
    exploitability guardrail).
20. Experiments: adaptation vs archetypes, opponent identification,
    distribution shift, exploration-vs-exploitation, overfitting-vs-sample-size,
    rake sensitivity.
21. Risk analytics: return metrics, Kelly, bankroll simulation, risk of ruin.
22. Performance profiling + optimization benchmark.
23. README finalization with benchmark tables from real runs.
24. RESEARCH.md writeup + final demo (`python -m poker_alpha.demo`).
