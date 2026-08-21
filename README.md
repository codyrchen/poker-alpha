# PokerAlpha

Game-theoretic poker solver and adaptive exploitation engine — a quantitative
research project studying the tradeoff between **equilibrium robustness** and
**opponent-specific exploitation** in imperfect-information games.

> **Core research question:** How much equilibrium robustness should a poker
> agent sacrifice in order to exploit statistically detected weaknesses in an
> opponent?

This is an offline simulation and research project. It does not connect to,
automate, or interact with any real poker platform.

> **📄 [RESEARCH.md](RESEARCH.md)** — the full quantitative writeup: the whole
> project organized around that question, with every measured result, the
> negative results, and the limitations. Start there for the research
> narrative; this README is the reference documentation.

## Status

Phases 1–18 of the [development plan](TODO.md) are complete and verified
(144 tests):

- **Kuhn Poker** implemented as an extensive-form game (states, chance nodes,
  information sets, utilities).
- **Leduc Poker** — public card, two betting rounds with raises and a raise
  cap, pot-based fold/showdown utilities; converges to the known game value.
- **Vanilla CFR** with regret matching, cumulative/average strategies.
- **CFR+** with regret clipping, alternating updates, and linear averaging.
- **External-sampling MCCFR** with seeded, deterministic sampling.
- **Exploitability evaluation** via a true imperfect-information best response.
- **Convergence experiments** with reproducible results.
- **Card engine + hand evaluator** — 52-card integer representation, all nine
  hand categories as lexicographically comparable tuples (wheel included),
  best-of-seven; 25 tests.
- **Monte Carlo equity engine** — seeded showdown-equity estimation against
  uniform or weighted opponent ranges with blocker handling. Validated against
  known values: AA vs random 85.1% (reference 85.2%), 72o 34.1% (≈34.6%).
- **Abstracted heads-up NL Hold'em** — 100 BB stacks, four streets, pot-relative
  sizing, sampled chance, chip-conservation-checked random playouts.
- **Opponent modeling** — six archetypes as multiplicative tilts of the
  equilibrium; a stationary Bayesian belief identifying the opponent from
  public actions only; a risk-constrained adaptive strategy
  (confidence-weighted λ blend, exact exploitability guardrail); and a
  recency-aware belief (exponential forgetting) for non-stationary opponents,
  with the regime-change tradeoff it buys — and costs — measured directly. See
  [Opponent modeling: identification, adaptation, and its limits](#opponent-modeling-identification-adaptation-and-its-limits)
  below for the numbers, including an honest negative result and its fix.
- **Risk analytics** — return metrics with bootstrap CIs, Kelly criterion,
  bankroll / risk-of-ruin simulation.

Current measured results (`python experiments/kuhn_convergence.py --iterations 100000 --seed 42`):

| quantity | measured | theory |
| --- | --- | --- |
| game value (player 0) | −0.055554 | −1/18 ≈ −0.055556 |
| exploitability after 100k iterations | 6.3 × 10⁻⁴ chips | → 0 at Nash |
| King facing bet: call | 100% | 100% (dominant) |
| Jack facing bet: fold | 100% | 100% (dominant) |
| Queen facing bet: call | 0.336 | 1/3 |

![CFR convergence on Kuhn Poker](results/figures/kuhn_convergence.png)

Solver comparison on Kuhn
(`python experiments/cfr_comparison.py --iterations 100000 --seed 42`),
measured on the same machine, exploitability in chips:

| algorithm | game | iterations | runtime | it/s | final exploitability |
| --- | --- | --- | --- | --- | --- |
| CFR | Kuhn | 100k | 27.9 s | 3,587 | 6.3 × 10⁻⁴ |
| CFR+ | Kuhn | 100k | 53.6 s | 1,865 | 2.1 × 10⁻⁶ |
| MCCFR (external sampling) | Kuhn | 100k | 12.3 s | 8,143 | 2.3 × 10⁻³ |

Two readings worth internalizing:

- **CFR+ converges near O(1/T)** versus CFR's O(1/√T) — ~300× lower
  exploitability here — at ~1.9× cost per iteration (an iteration is two
  alternating traversals, one per player).
- **MCCFR iterations are 2.3× cheaper than CFR's even on Kuhn**, but its
  sampled convergence is noisier, and on a ~50-node tree cheap iterations
  can't win: CFR+ dominates per unit wall-clock. Sampling pays off as the game
  grows — that scaling is what the Leduc experiment measures.

![CFR vs CFR+ vs MCCFR convergence](results/figures/cfr_comparison.png)
![CFR vs CFR+ vs MCCFR by wall-clock](results/figures/cfr_comparison_time.png)

### Scaling up: Leduc Poker

Leduc (`python experiments/leduc_convergence.py --iterations 1000 --seed 42`)
is ~75× larger than Kuhn by decision nodes (3,780 vs ~50; 288 rank-merged
information sets). Measured results:

| algorithm | game | iterations | runtime | it/s | final exploitability |
| --- | --- | --- | --- | --- | --- |
| CFR | Leduc | 1k | 50.7 s | 19.7 | 1.6 × 10⁻² |
| CFR+ | Leduc | 1k | 92.2 s | 10.8 | 2.5 × 10⁻⁴ |
| MCCFR | Leduc | 200k | 106.0 s | 1,886 | 2.9 × 10⁻² |

- Full-traversal cost scaled with the tree: CFR went from 3,587 it/s on Kuhn
  to 19.7 it/s on Leduc (~180× slower per iteration), while MCCFR's sampled
  iterations only slowed ~4× (8,143 → 1,886 it/s). The *relative* cheapness of
  sampling grew from 2.3× to ~96×.
- CFR+ again dominates per iteration (~64× lower exploitability than CFR at
  1k) and converges to the known game value (measured −0.0855 vs −0.0856).
- **Honest negative result:** even at matched wall-clock, MCCFR still trails
  exact CFR on Leduc (4.7 × 10⁻² vs 1.6 × 10⁻² at ~50 s) — sampling variance
  dominates on a game that full traversal can still comfortably sweep. MCCFR's
  real payoff is games where a full traversal is infeasible *per se* (the
  abstracted Hold'em phase), not mid-sized games.

![CFR vs CFR+ on Leduc](results/figures/leduc_convergence_exact.png)

An implementation note worth knowing for interviews: CFR+'s regret clip must be
applied to an information set's **total** regret per iteration, not per visited
history. An information set spans several histories (in Kuhn, "hold the Jack"
spans two opponent cards), and clipping between visits biases the update — with
that bug, CFR+ degraded to CFR's O(1/√T) rate; buffering the per-iteration
deltas and clipping once restored O(1/T). The commit history preserves both
measurements.

### Opponent modeling: identification, adaptation, and its limits

The core research question — how much equilibrium robustness to trade for
opponent-specific exploitation — is tested directly on Leduc, where every
strategy's exact EV and exploitability are computable, not just estimated.
Four experiments (`experiments/adaptation_vs_archetypes.py`,
`opponent_identification.py`, `overfitting_vs_sample_size.py`,
`regime_change.py`; seed 42, 300-iteration CFR+ equilibrium):

**How fast can the opponent be identified?** A Bayesian posterior over the six
archetypes, updated hand-by-hand from public actions only (hero plays static
equilibrium; 50 repeats/archetype):

| archetype | correct by 5 hands | by 50 hands | by 100 hands |
| --- | --- | --- | --- |
| maniac | 78% | 96% | 100% |
| calling_station | 40% | 92% | 94% |
| bluff_heavy (subtlest tilt) | 18% | 72% | 84% |

![Opponent identification accuracy](results/figures/opponent_identification_accuracy.png)

**Does adaptation pay?** Three heroes, 3000-hand matches: static equilibrium,
an oracle best response (knows the true opponent strategy, uncapped — an EV
ceiling, not a realizable agent), and the online adaptive agent (posterior
mixture, refreshed every 20 hands, capped to an exploitability budget ε=0.1):

| opponent | equilibrium | oracle BR (exploitability) | adaptive (exploitability) |
| --- | --- | --- | --- |
| maniac | −9.0 | +77.8 (2.42) | +10.9 (≤0.10) |
| calling_station | −10.1 | +38.2 (1.82) | +3.1 (≤0.10) |
| bluff_heavy | −4.9 | +47.6 (2.60) | −1.7 (≤0.10) |

(chips/100 hands; oracle exploitability in chips, ~1000× the equilibrium's
own ~0.002.) The adaptive agent recovers a real slice of the oracle's edge at
a small, bounded exploitability cost — but at 3000 hands the bootstrap CIs on
these numbers are ~±17 chips/100, wide enough that most individual
adaptive-vs-equilibrium deltas above are not significant on their own.

![Adaptive vs equilibrium vs oracle](results/figures/adaptation_vs_archetypes.png)

Sweeping the exploitability budget ε against the maniac confirms the
*mechanism* works exactly as designed — mean applied λ rises monotonically
with ε (0 → 0.013 → 0.032 → 0.059 → 0.16 → 0.31 as ε: 0 → 1.0, matching the
uncapped ceiling computed directly from the posterior's deviation estimate) —
but a single 2000-hand run's realized EV at each ε is dominated by Leduc's
per-hand variance, not ε. Confirming the *EV* payoff of the tradeoff needs the
repeats/bootstrap approach above, not a one-shot sweep.

**Does the exploitability guardrail actually prevent overfitting to noise?**
Freezing a strategy after only *n* hands of belief formation and scoring its
*exact* EV (`profile_value`, no evaluation noise; 15 repeats/point) shows the
posterior's own confidence gating already blocks most early overfitting — at
5–20 hands, low confidence caps λ regardless of the guardrail, so guarded and
unguarded strategies both stay near equilibrium. Once confidence saturates
(60+ hands), the guardrail's effect is visible: against maniac at 1000 hands,
guarded settles at 0.031 chips/hand vs unguarded's 0.184 (oracle 1.075) — most
of the unguarded edge traded away for a much smaller exploitability
footprint. In worst-case draws the guardrail bounds tail losses when it binds
(calling_station @ 20 hands: guarded_min −0.020 vs unguarded_min −0.034) and
is a no-op, not a floor, when it doesn't.

**Honest negative result: regime change.** The opponent plays maniac for 2000
hands, then switches to nit for 2000 more, with no signal to the hero. The
posterior locks onto maniac early in the first regime — then, after the
switch, it stays *confidently wrong*: the MAP estimate is never correct at
any post-switch checkpoint, and posterior mass on the true new archetype
stays numerically ~0 (below 1e-319) across the entire remaining 2000 hands,
in all 3 repeats. `ArchetypeBelief` accumulates log-likelihood over every hand it has
ever seen with no decay: a sound assumption for a stationary opponent, and
actively harmful for a non-stationary one — once confidence saturates, old
evidence permanently outvotes new evidence forever.

### Fixing it: a recency-aware belief, and what it costs

`ArchetypeBelief` now takes a `decay` parameter: exponential forgetting of the
accumulated log-likelihood (`LL_t = decay·LL_{t-1} + ll_t`, effective memory
horizon ≈ 1/(1−decay) hands). `decay=1.0` is the exact stationary baseline
above — unchanged, still the default, every existing call site and test
untouched. This is not a free upgrade: forgetting old evidence also means a
decaying belief has less protection against noise, so it can *misidentify a
stationary opponent* just from an unlucky run of hands. `regime_change.py`
now measures both sides on the identical maniac→nit switch (baseline and
`decay=0.995` fed the exact same paired hand sequence) and on a
stationary-opponent control (no switch at all, 30 repeats, decay swept from
1.0 to 0.98):

| | baseline (decay=1.0) | recency (decay=0.995) |
| --- | --- | --- |
| detection delay (hands post-switch to a *sustained* correct MAP estimate) | never, in 3/3 repeats | 380–500, in 3/3 repeats |
| confidence during recovery | >0.99 at 96% of post-switch checkpoints, never below 0.87 — *confidently* wrong | falls to 0.76, then re-saturates |
| false-switch rate, stationary maniac (30 reps, post-warmup) | 0.02% | 0.0% |
| false-switch rate, stationary bluff_heavy (30 reps, post-warmup) | 0.71% | 1.39% |

![Posterior mass on the true archetype: baseline never recovers, recency does](results/figures/regime_change_posterior_true.png)
![Confidence: baseline stays near 1.0 on the stale label through the switch, while recency dips visibly as it recovers](results/figures/regime_change_confidence.png)

decay=0.995 is a good point on the curve, not a free lunch: sweeping decay
down to 0.98 on the stationary control shows *why* it isn't pushed lower —
false switches on bluff_heavy (the hardest archetype to read) jump to 10.8%
of checkpoints (8.6 flip episodes per 3000-hand repeat), trading faster
hypothetical recovery for materially worse stability against an opponent that
never actually changed.

![False-switch rate rises as decay drops](results/figures/regime_change_control_flip_rate.png)

Under this experiment's ε=0.1 exploitability guardrail (from the adaptation
experiment above), the exact EV (`profile_value`) of both baseline's and
recency's frozen strategy tracks close to equilibrium — well short of the
oracle ceiling either way — so the guardrail itself absorbs most of the
downside of misidentification at this budget; a looser ε would make getting
the identification right (or wrong) matter more.

![EV lost after the switch, baseline vs recency, against the equilibrium and oracle bounds](results/figures/regime_change_ev.png)

Upcoming (see [TODO.md](TODO.md)): exploration-vs-exploitation and
rake-sensitivity experiments, performance profiling, and a final RESEARCH.md
writeup.

## The math so far (no CFR background assumed)

**Extensive-form games & information sets.** Poker is a sequential game where
players cannot see each other's cards. All game states that look identical to
the acting player (same own cards, same public history) form an *information
set*; a strategy must choose the same action distribution everywhere inside one
— that is what "not seeing the opponent's cards" means formally.

**Regret.** After playing, compare what you earned with what you *would* have
earned by always picking some action `a` at an information set. That difference
is the regret for `a`. *Counterfactual* regret weights this by the probability
that the other players and chance even let you reach that information set.

**Regret matching.** Next iteration, play each action with probability
proportional to its accumulated *positive* regret:

```
σ(a) = max(R(a), 0) / Σ_b max(R(b), 0)      (uniform if all regrets ≤ 0)
```

**Why CFR converges.** Regret matching guarantees average regret grows like
O(√T), so average regret per iteration → 0. In two-player zero-sum games, a
profile in which neither player has positive average regret is a Nash
equilibrium — and the guarantee applies to the **time-averaged** strategy, which
is why the solver tracks cumulative strategy sums and reports the average (the
current iterate oscillates; the average converges).

**Exploitability** measures the distance from equilibrium: how much a
best-responding adversary could win against your strategy, averaged over both
seats. Zero exactly at Nash. Our best response respects information sets (it
does not peek at hidden cards), computed by resolving information sets
deepest-first with counterfactual reach weights.

## Performance engineering

Profile first, optimize second, prove correctness unchanged, then re-measure.
All numbers below come from `experiments/performance_benchmark.py` (7 timed
repetitions after a warmup, `time.perf_counter`) and live in
`results/data/performance_comparison.csv`.

### What profiling identified

`cProfile` on 20,000 preflop equity simulations put **94.6% of runtime inside
`evaluate_best`**. The cause was algorithmic, not micro: evaluating seven
cards by taking `max` over every five-card subset costs C(7,5) = 21
`evaluate_five` calls, so one simulated hand (hero + villain) ran 42 of them —
840,000 calls for the 20k-simulation workload. Monte Carlo sampling, the
part that *looks* expensive, was negligible by comparison.

### What was optimized

1. **Direct 7-card evaluation.** `evaluate_best` now derives the hand
   directly from rank/suit histograms and a 13-bit rank mask, testing
   categories in descending order and returning on the first match. Straights
   use the bit trick `m = r & r>>1 & r>>2 & r>>3 & r>>4`, whose highest set
   bit is the straight's low card. This removes the 21× subset blowup
   entirely: 21 evaluations → 1.
2. **A pre-validated fast path.** With the evaluator fast, re-profiling
   showed `codes()` — the mixed-type (`Card` / `int` / `"As"`) normalizer —
   had risen to ~24% of equity runtime, doing 560k `isinstance` checks on
   values the sampler already produced as integers. `evaluate_best_codes`
   skips that normalization; `estimate_equity` calls it directly.

### Measured before/after

| workload | before | after | speedup |
| --- | --- | --- | --- |
| hand_eval_7card | 37,679 evals/s | 414,955 evals/s | **11.01×** |
| equity_flop_uniform | 17,108 sims/s | 135,370 sims/s | **7.91×** |
| equity_preflop_uniform | 17,874 sims/s | 126,444 sims/s | **7.07×** |
| equity_weighted_range | 15,068 sims/s | 74,764 sims/s | **4.96×** |
| hand_eval_5card | 831,594 evals/s | 810,468 evals/s | 0.97× |
| cfr_leduc / cfr_plus_leduc / mccfr_leduc | — | — | 1.02× / 1.02× / 1.01× |
| exploitability_leduc / profile_value_leduc | — | — | 0.96× / 0.95× |
| belief_update / leduc_match_simulation | — | — | 0.96× / 0.93× |

![Throughput before vs after](results/figures/performance_speedup.png)

### What did *not* improve, and why

The solver, exact-evaluation and opponent-modeling workloads are untouched:
none of them import the hand evaluator (Leduc compares ranks directly), so
they act as a **built-in control**. Their spread — 0.93× to 1.07× — is the
honest noise floor of this machine, and it is the reason no speedup below
roughly 1.1× anywhere in the table should be read as real. `hand_eval_5card`
is deliberately flat: `evaluate_five` was left exactly as it was, since it is
the reference semantics the fast path is verified against and it was never
the bottleneck.

`equity_weighted_range` gains least (4.96×) because a weighted range spends a
second `rng.choice` per simulation to draw the villain combo; with evaluation
no longer dominant, that NumPy call is a larger share of what remains. The
next optimization there would be batching the RNG draws — deliberately *not*
done, because it would change the per-seed random stream and therefore the
reproducibility contract below.

### Tradeoffs introduced

* `evaluate_best_codes` trades a little safety for speed: it skips input
  validation, so it is documented as a fast path for callers that already
  hold validated integer codes. `evaluate_best` remains the safe public entry
  point and is unchanged in behavior.
* The direct evaluator is denser code than a 21-way `max`. It is paid for by
  the equivalence tests, not by trust.
* No new dependency was added. `numba` remains an optional extra and is still
  unused; a ~11× algorithmic win made a JIT unnecessary.

### Correctness: the optimization is provably behavior-preserving

Speed gains do not count if results move. This one was pinned before and
after:

* **Exhaustive**: all **2,598,960** distinct five-card hands agree with
  `evaluate_five` exactly, and the category histogram reproduces the textbook
  frequencies (40 straight flushes, 624 quads, …, 1,302,540 high card).
* **Random**: 300,000 seven-card and 60,000 six-card hands agree with the
  original subset-max implementation — zero mismatches.
* **Adversarial**: two trips, three pairs, quads-plus-trips, steel wheel
  under a bigger flush, six- and seven-card flushes — the cases a direct
  evaluator typically gets wrong. All committed as tests.
* **End-to-end**: SHA-256 digests of hand-value streams, exact
  `estimate_equity` floats for identical seeds, and CFR/CFR+ average-strategy
  digests are **bit-identical** before and after. Because only the evaluation
  path changed and the RNG draw sequence was left alone, every seeded result
  in this repository still reproduces exactly.

### Complexity, measured rather than asserted

| component | cost | measured behavior |
| --- | --- | --- |
| Full-tree CFR / CFR+ iteration | O(\|tree\|) per traversal | Kuhn ~50 nodes → 8.4k it/s; Leduc 3,780 → 46 it/s, ~180× slower for a ~75× bigger tree |
| MCCFR iteration | O(depth) per sampled path | barely grows with tree size: 22.2k it/s Kuhn → 5.1k it/s Leduc (~4×), which is why sampling wins as games grow |
| Exact best response / exploitability | two full traversals + per-infoset argmax | 19 calls/s on Leduc — the reason the λ-guardrail bisection is the costliest part of the adaptive agent |
| Hand evaluation (7 cards) | was O(C(7,5)) = 21 sub-evaluations; now O(1) histogram | 37.7k → 415k evals/s |
| Monte Carlo equity | O(simulations) evaluations; error O(1/√n) | 17.1k → 135k sims/s, so a fixed latency now buys ~7× the samples and ~√7 ≈ 2.6× tighter error |

The memory-vs-compute tradeoff was considered and **rejected**: a full
precomputed 7-card lookup table would make evaluation a single array index,
but C(52,7) = 133,784,560 entries is hundreds of MB and would need building
or shipping. The histogram evaluator gets most of the win with zero
preprocessing and no memory footprint — the right point on that curve for a
research codebase.

### Compute budget vs decision quality

A decision system rarely runs to convergence — it gets a budget.
`experiments/compute_quality_tradeoff.py` trains each solver in chunks until a
wall-clock budget is spent (evaluation clock stopped, so measuring never eats
the budget) and scores the result by exploitability. This is an
imperfect-information-game analogue of a latency budget, not a production
trading benchmark.

The headline finding is that **the best algorithm depends on the budget**:

| budget | CFR | CFR+ | MCCFR | best |
| --- | --- | --- | --- | --- |
| 0.05 s | 0.972 | 2.058 | 1.320 | CFR |
| 0.25 s | 0.279 | 1.011 | 0.637 | CFR (3.6× better than CFR+) |
| 0.5 s | 0.182 | 0.358 | 0.390 | CFR |
| 1 s | 0.105 | 0.096 | 0.226 | CFR+ (crossover) |
| 5 s | 0.034 | 0.0095 | 0.082 | CFR+ |
| 20 s | 0.0176 | 0.00093 | 0.045 | CFR+ (19× better than CFR) |

(exploitability in chips; lower is better)

![Strategy quality per unit compute](results/figures/compute_quality_tradeoff.png)

Below about half a second, plain CFR wins: CFR+ spends two alternating
traversals per iteration, so it buys roughly half the iterations, and its
regret clipping and linear averaging need enough iterations before their
asymptotically better rate pays off. Past the ~1 s crossover the asymptotics
take over and CFR+'s advantage compounds to 19× by 20 s. MCCFR trails at every
budget on a game this small — the same honest negative already documented for
Leduc: sampled iterations are only worth their variance when a full traversal
is infeasible.

The equity engine shows the other half of the tradeoff. Monte Carlo error
falls as 1/√n, so accuracy is bought at quadratic cost — and measured RMS
error over 12 seeds tracks that law closely:

| latency | simulations | RMS error |
| --- | --- | --- |
| 7.9 ms | 1,000 | 1.62 × 10⁻² |
| 78 ms | 10,000 | 3.39 × 10⁻³ |
| 2.3 s | 300,000 | 7.9 × 10⁻⁴ |

![Equity accuracy per millisecond](results/figures/compute_quality_equity.png)

This is where the optimization cashes out in decision quality rather than
vanity throughput: at a fixed 10 ms budget the old evaluator afforded ~171
simulations, the new one ~1,354 — and since error scales as 1/√n, that is
**≈2.8× tighter error bars for the same latency**.

## Running

```bash
pip install -e ".[dev]"     # or: pip install -r requirements.txt
pytest                       # 144 tests
python experiments/kuhn_convergence.py --iterations 100000 --seed 42
python experiments/adaptation_vs_archetypes.py --seed 42
python experiments/opponent_identification.py --seed 42
python experiments/overfitting_vs_sample_size.py --seed 42
python experiments/regime_change.py --seed 42
python experiments/performance_benchmark.py --label baseline
python experiments/compute_quality_tradeoff.py --seed 42
```

Experiments take command-line arguments (`--iterations`/`--hands`, `--seed`,
`--outdir`) and write raw CSV data to `results/data/` and figures to
`results/figures/`. All randomness is seeded for reproducibility.

## Repository structure

```
poker_alpha/
  games/      extensive-form game definitions (base interface, Kuhn; Leduc & Hold'em later)
  solvers/    CFR (+ CFR+/MCCFR later) and strategy evaluation (EV, best response, exploitability)
  poker/      card engine, hand evaluator, equity simulation (later phases)
  opponent/   archetypes, statistics, Bayesian modeling, exploitation (later phases)
  risk/       bankroll, Kelly, risk metrics (later phases)
  utils/      seeding, plotting
experiments/  runnable, parameterized experiments
tests/        pytest suite
results/      generated data and figures (never hand-entered)
```

## Design principles

- Solvers depend only on a minimal `Game` interface, so one CFR implementation
  trains on every game in the project.
- Math is kept visible: `regret_matching(regrets: np.ndarray)` is a function,
  not a framework.
- All reported numbers come from actual runs; nothing in this README or the
  research notes is hand-entered.
- Results carry uncertainty wherever sampling is involved (later phases add
  bootstrap confidence intervals for match results).

## License

MIT
