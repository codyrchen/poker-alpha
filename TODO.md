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

14. Risk analytics (`risk/`) — metrics.py (mean/variance/downside/Sharpe-like
    with caveats, max drawdown, VaR/CVaR, win rate, bb/100, percentile
    bootstrap CIs, RiskReport), kelly.py (f*, log-growth, fractional-Kelly
    Monte Carlo showing over-betting lowers median wealth and deepens
    drawdowns), bankroll.py (risk-of-ruin, percentiles, drawdown
    distributions). 17 tests (132 total).

15. Headline experiments (`experiments/adaptation_vs_archetypes.py`,
    `opponent_identification.py`, `overfitting_vs_sample_size.py`,
    `regime_change.py`) + figures. All four use the existing `opponent/` and
    `risk/` libraries as-is (only new code: `save_grouped_bar_chart` in
    `utils/plotting.py` and a small experiment-only helper module,
    `experiments/_opponent_common.py`). Measured (seed 42, 300-iteration CFR+
    equilibrium):
    - **Identification** (50 repeats/archetype, hero static at equilibrium):
      maniac's MAP estimate is correct 78% of the time by 5 hands, 96% by 50,
      100% by 100; bluff_heavy (the subtlest tilt) is the slowest, 18% by 5
      hands, 72% by 50, 100% only by 800.
    - **Adaptation vs archetypes** (3000-hand matches, refresh every 20 hands,
      exploitability budget ε=0.1): the oracle best response (knows the
      opponent exactly, uncapped) beats equilibrium by a wide, exploitability-
      expensive margin (e.g. maniac +77.8 chips/100 at 2.4 chips
      exploitability vs equilibrium's 0.002); the realized online adaptive
      agent captures a real but much smaller edge (maniac +10.9 vs
      equilibrium's -9.0) while its exploitability stays ≤ the 0.1 budget
      throughout — but at 3000 hands the bootstrap CIs on chips/100 are ~±17
      chips wide, so most single-match adaptive-vs-equilibrium deltas are not
      individually significant; this needs many repeats, not one long match,
      to resolve tightly.
    - **ε (robustness budget) sweep vs maniac**: mean applied λ scales
      monotonically and exactly as designed with the budget (0 → 0.013 →
      0.032 → 0.059 → 0.16 → 0.31 as ε: 0 → 1.0, matching the
      confidence-1/no-cap ceiling of λ≈0.313 computed directly from
      `deviation_magnitude`); realized EV from a single 2000-hand run at each
      ε does not — it is dominated by Leduc's per-hand variance at this
      sample size. The mechanism is verified; measuring its EV payoff
      precisely needs repeats/bootstrap, which the fixed-ε table above
      supplies instead.
    - **Overfitting vs sample size** (exact EV via `profile_value`, no
      simulation noise; only belief formation is stochastic; 15 repeats):
      at 5-20 hands of evidence, low belief confidence caps λ regardless of
      the exploitability guardrail, so guarded and unguarded frozen
      strategies both stay close to equilibrium — the confidence gate alone
      already blocks most catastrophic early overfitting. Once confidence
      saturates (60+ hands) the guardrail's effect becomes visible: e.g.
      maniac at 1000 hands, guarded settles at 0.031 chips/hand vs
      unguarded's 0.184 (oracle ceiling 1.075) — the guardrail trades most of
      the unguarded edge away for a much smaller exploitability footprint.
      Worst-case draws show the guardrail bounding tail risk when it binds
      (calling_station @ 20 hands: guarded_min -0.020 vs unguarded_min
      -0.034) and doing nothing when confidence hasn't caught up yet (it's a
      no-op, not a floor).
    - **Regime change, honest negative result**: opponent plays maniac for
      2000 hands then switches to nit for 2000 more, hero none the wiser.
      `ArchetypeBelief`'s posterior correctly locks onto maniac by ~180
      hands — then, post-switch, confidence stays pegged at ~1.0 (on the now
      *wrong* label) and posterior mass on the true new archetype (nit)
      measures numerically ~0 for the entire remaining 1980 post-switch
      hands, in all 3 repeats, reaching only ~1e-280 by hand 3980. Because
      the posterior accumulates log-likelihood over every hand ever observed
      with no decay, old evidence permanently outvotes new evidence once
      confidence has saturated — a sound design for a stationary opponent,
      actively harmful for a non-stationary one. Left as a documented
      limitation (see Future: recency-windowed or exponentially-discounted
      belief) rather than patched speculatively.

## Future

13. Equity-bucket card abstraction; document information loss.
14. Real-time subgame solving; compute-vs-quality experiment.
15. Recency-windowed or exponentially-discounted `ArchetypeBelief` variant to
    handle non-stationary opponents (motivated by the regime-change negative
    result above); re-run `regime_change.py` against it.
16. Additional experiments: exploration-vs-exploitation, rake sensitivity.
17. Performance profiling + optimization benchmark (equity engine currently
    ~7k sims/s — the flagged optimization target).
18. README finalization with benchmark tables from real runs.
19. RESEARCH.md writeup + final demo (`python -m poker_alpha.demo`).
