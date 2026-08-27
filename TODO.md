# PokerAlpha — Development Plan

> **Status: complete.** Phases 1–19 are done, tested, and committed. This file
> is retained as development history — it records what was built in what order,
> what each phase measured, and which assumptions the experiments falsified
> along the way. The items under *Optional extensions* are ideas, not
> outstanding work.
>
> Start at [README.md](README.md) for the overview, [RESEARCH.md](RESEARCH.md)
> for the research narrative, or `python -m poker_alpha.demo` for a 12-second
> tour.

Working order followed an incremental plan: each phase was implemented
smallest-correct-first, tested, smoke-run, and committed before the next began.

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
    11 tests. Validated: AA 85.1% (known 85.2%), 72o 34.1% (known ~34.6%).
    (An earlier note here estimated ~7k sims/s; phase 17 measured the real
    figure at 17.1k sims/s before optimization and 135k after.)

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
      `ArchetypeBelief`'s posterior correctly locks onto maniac early in the
      first regime — then, post-switch, it stays confidently wrong: the MAP
      estimate is never correct at any post-switch checkpoint and posterior
      mass on the true new archetype (nit) stays numerically ~0 (below
      1e-319) for the entire remaining 2000 hands, in all 3 repeats. Because the
      posterior accumulates log-likelihood over every hand ever observed
      with no decay, old evidence permanently outvotes new evidence once
      confidence has saturated — a sound design for a stationary opponent,
      actively harmful for a non-stationary one. Fixed in phase 16 below,
      whose paired baseline-vs-recency run reproduces this baseline behavior
      and supplies the numbers quoted here (the phase-15 run's own CSV was
      superseded by it; see git history at `e97a0de` for the original).

16. Recency-aware `ArchetypeBelief` (`opponent/beliefs.py`): a `decay`
    parameter (default 1.0, exactly the phase-15 stationary baseline —
    backward compatible, no existing call site or test changed) applies
    exponential forgetting to the accumulated log-likelihood before each
    hand's update (`LL_t = decay * LL_{t-1} + ll_t`; effective memory
    horizon ≈ 1/(1-decay) hands). A rolling evidence window was the other
    natural design; exponential forgetting was chosen as an O(1) per-hand
    update to the existing scalar accumulator, no history buffer needed.
    3 new tests (135 total).

    `experiments/regime_change.py` rewritten to compare baseline (decay=1.0)
    against recency (decay=0.995, chosen from the control sweep below) on
    the identical maniac→nit switch, paired on one hand sequence (hero fixed
    at equilibrium) so both beliefs see exactly the same evidence. Measured
    (seed 42, 3 repeats):
    - **Detection delay**: baseline never recovers within the 2000 post-switch
      hands (confirms phase 15). Recency recovers a *sustained* correct MAP
      estimate 380–500 hands after the switch, in all 3 repeats.
    - **Confidence**: baseline's confidence stays above 0.99 at 96% of
      post-switch checkpoints (on the now-stale label), never dipping below
      0.87 — it is not merely wrong, it is *confidently* wrong. Recency's
      confidence falls to 0.76 during the ~500-hand recovery window before
      re-saturating near 1.0: a visible "I'm confused" signal the baseline
      never really produces. (Per-checkpoint means over 3 repeats, which is
      what `regime_change_confidence.png` plots.)
    - **EV lost**: with the ε=0.1 exploitability guardrail from phase 15
      active, both baseline's and recency's exact EV (`profile_value`) track
      close to equilibrium — well short of the oracle best response either
      way. The guardrail itself bounds most of the downside of misidentifying
      the opponent at this budget; a looser ε would make correct
      identification matter more (see phase 15's ε-sweep finding).
    - **False-switch / noise sensitivity** (stationary-opponent control, no
      actual regime change, 30 repeats, decay ∈ {1.0, 0.999, 0.995, 0.99,
      0.98} × {maniac, bluff_heavy}): flip rate rises monotonically as decay
      drops. At decay=0.995 it stays close to the stationary baseline's own
      rate (maniac 0% vs 0.02%; bluff_heavy 1.4% vs 0.7%) while still
      recovering from a real switch in ~450 hands. At decay=0.98, false
      switches on bluff_heavy jump to 10.8% of checkpoints (8.6 flip episodes
      per 3000-hand repeat) — recovering faster from a real switch, but far
      noisier against a stationary opponent. decay=0.995 is a good point on
      that curve, not a free lunch: the tradeoff is real and measured, not
      assumed.

17. Performance profiling and optimization (`experiments/performance_benchmark.py`,
    `experiments/compute_quality_tradeoff.py`). Measure first, optimize only
    what profiling justifies, prove results unchanged, re-measure.
    - **Baseline + profiling.** 15 workloads benchmarked (7 timed reps after
      warmup, mean/median/std/min/max + throughput). `cProfile` on 20k equity
      simulations put **94.6% of runtime in `evaluate_best`**: evaluating 7
      cards as `max` over all C(7,5)=21 subsets meant 42 `evaluate_five`
      calls per simulated hand (840k calls for the workload). Sampling —
      the part that looks expensive — was negligible. The old note claiming
      ~7k sims/s was wrong; the real baseline was 17.1k.
    - **Optimization 1: direct 7-card evaluation.** `evaluate_best` now works
      from rank/suit histograms and a 13-bit rank mask (straights via
      `m = r & r>>1 & r>>2 & r>>3 & r>>4`), testing categories in descending
      order. 21 evaluations → 1.
    - **Optimization 2: pre-validated fast path.** Re-profiling showed
      `codes()` had risen to ~24% of equity runtime (560k `isinstance` calls
      on values the sampler already emits as ints). `evaluate_best_codes`
      skips normalization; `estimate_equity` calls it directly.
    - **Measured** (back-to-back runs, same machine): hand_eval_7card
      **11.0×** (37.7k → 415k evals/s), equity_flop **7.9×** (17.1k → 135k
      sims/s), equity_preflop **7.1×**, equity_weighted_range **5.0×**.
    - **Correctness preserved exactly.** All 2,598,960 five-card hands match
      `evaluate_five` (category histogram reproduces textbook frequencies);
      300k 7-card and 60k 6-card hands match the original subset-max
      implementation; adversarial cases (two trips, three pairs, steel wheel
      under a bigger flush, 6-/7-card flushes) committed as tests. SHA-256
      digests of hand-value streams, exact `estimate_equity` floats for
      identical seeds, and CFR/CFR+ strategy digests are **bit-identical**
      before and after — the RNG stream was deliberately left untouched.
      9 new tests (144 total).
    - **Honest non-results.** Solver, exact-evaluation and opponent-modeling
      workloads are unchanged (0.93×–1.07×) — none import the evaluator, so
      they serve as a built-in control whose spread *is* this machine's noise
      floor (nothing under ~1.1× is meaningful). `evaluate_five` was left
      alone on purpose. Batching the equity RNG draws was rejected: it would
      break per-seed reproducibility. A precomputed C(52,7) lookup table was
      rejected on memory grounds. No dependency added; `numba` still unused.
    - **Compute budget vs decision quality.** Solver quality per wall-clock
      budget shows the best *algorithm* depends on the budget: plain CFR beats
      CFR+ below ~0.5 s (3.6× better at 0.25 s, since CFR+ spends two
      traversals per iteration and its averaging needs iterations to pay
      off), they cross near 1 s, and CFR+ is 19× better by 20 s. MCCFR trails
      at every budget on a game this small. On the equity side, measured RMS
      error over 12 seeds tracks the 1/√n law; the speedup converts to
      accuracy: at a fixed 10 ms budget the engine went from ~171 to ~1,354
      simulations, i.e. **≈2.8× tighter error bars for the same latency**.

18. Quantitative research writeup (`RESEARCH.md`) — the whole project
    reorganized around its central question ("how much equilibrium robustness
    should an agent sacrifice to exploit statistically detected deviations,
    under noise, non-stationarity, and a compute budget?") rather than as a
    feature list. ~5.3k words of prose, 7 figures, 14 sections: game-theoretic
    foundation, solver experiments, opponent modeling, adaptive exploitation,
    estimation risk, the distribution-shift negative result and its fix,
    performance engineering, the compute/quality tradeoff, risk, limitations,
    findings, and reproducibility.

    Documentation-only: no source change, test count unchanged at 144. Every
    numerical claim was verified programmatically against the committed CSVs
    before commit (solver tables, identification curves, adaptation and
    ε-sweep, overfitting, regime change, control sweep, performance
    comparison, equity accuracy, and the Kelly table recomputed from
    `poker_alpha.risk`), and all equations were checked against the
    implementations they describe. Two cross-cutting observations that only
    became visible when the results were assembled in one place:
    - **Detectability and exploitability are close to unrelated.** Across the
      five non-balanced archetypes, identification accuracy at 50 hands
      correlates with total-variation deviation at r = 0.88 but with
      exploitability at only r = 0.28 (n = 5, indicative). `bluff_heavy` is
      the hardest archetype to identify yet the second most exploitable.
    - **The guardrail, not confidence, is what binds** in the headline
      adaptation runs: at ε = 0.1 the applied λ averages 0.059 against a
      confidence-implied ceiling of 0.313.

19. Final demo and repository polish. `poker_alpha/demo.py` — one
    deterministic command (`python -m poker_alpha.demo`, ~12s) that walks the
    whole research narrative: equilibrium baseline on Kuhn and Leduc solved
    live; Bayesian identification of a hidden `maniac` from public actions
    only, with the posterior printed at log-spaced checkpoints; the same
    machinery against `bluff_heavy` to show that detectability and
    exploitability are separate axes; the λ tradeoff curve (exact EV against
    exploitability) with the guardrail's actual choice marked; and the
    regime-change and performance results read from committed CSVs. Every
    section is labelled *computed live* or *committed results*, and no value
    is hard-coded — the CSV loaders and the sustained-recovery logic are
    covered by 24 new tests (168 total).

    README rewritten for a first-screen read: what the project is, four
    navigation links, five headline measurements, quick start, the research
    question, key findings, four selected figures, and limitations up front —
    with the detailed results kept below and RESEARCH.md as the narrative.
    Repository-structure and design-principles sections de-staled (they still
    described modules as "later phases"), and the experiment list split by
    cost so nobody assumes the expensive runs are instant.

## Optional extensions

Ideas, not outstanding work — the project above is complete without them.

- Equity-bucket card abstraction, documenting the information loss.
- Real-time subgame solving.
- Exploration-vs-exploitation and rake-sensitivity experiments.
- Scaling the opponent-modeling results to the abstracted Hold'em engine,
  where MCCFR's sampling advantage should finally dominate (the Leduc
  measurements say it does not at that size).
