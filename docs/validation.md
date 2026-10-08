# Validation status (Phase 25)

Machine-readable results: [`results/validation/holdem_platform_validation.json`](../results/validation/holdem_platform_validation.json)
(regenerate with `python experiments/phase25_validation.py --runs-dir RUNS`,
where `RUNS` holds the output of `experiments/holdem_mccfr_validation.py`).
Every number below is copied from that file.

## Summary

| status | item |
| --- | --- |
| **VALIDATED** | existing Kuhn/Leduc research regressions (pinned digests, now cross-platform; see "Reproducibility migration") |
| **VALIDATED** | rules-engine invariants (chip conservation, side pots, reopening rules; property tests on random 2–9 seat hands) |
| **VALIDATED** | checkpoint exact resume (CFR, CFR+, MCCFR incl. RNG state) |
| **VALIDATED** | sampled chance (MCCFR on Hold'em without chance enumeration) |
| **VALIDATED** | same-corpus abstraction key counts and the "abstract keys ≤ raw keys" invariant |
| **VALIDATED** | synthetic observer fixtures (synthetic only) |
| **VALIDATED** | analytical decision invariants (10 invariants + 2 split-pot regressions) |
| **PARTIALLY VALIDATED** | perfect-recall properties of `HoldemBucketEncoder` (corpus audit, not a proof) |
| **PARTIALLY VALIDATED** | MCCFR learning behaviour (measured; it is mostly discovery) |
| **PARTIALLY VALIDATED** | multiway recommendation quality (exact-formula invariants only) |
| **NOT VALIDATED** | a converged heads-up Hold'em strategy |
| **NOT VALIDATED** | a low-exploitability strategy (exploitability is not computed) |
| **NOT VALIDATED** | equilibrium-quality real Hold'em play |
| **NOT VALIDATED** | real PokerNow recognition accuracy (0 real fixtures) |
| **NOT VALIDATED** | profitability |
| **NOT VALIDATED** | calibration of range priors / behaviour models against real opponents |

## 1. Same-corpus abstraction measurement

Earlier reports compared infoset counts from separate MCCFR runs (raw 49,050,
toy 23,940, bucket 77,872). Those runs followed different sampled
trajectories and the bucket run used a different bet menu, so the
comparison measured nothing about compression and is withdrawn.

Method (`poker_alpha/validation/abstraction_audit.py`): one corpus from 2,000
seeded random-legal-policy hands (fold 0.15 / check-call 0.50 / bets share
the rest; seed 0; bet menu 33/75/150% pot, 100 BB), exact duplicate raw
states removed, then every encoder applied to the same frozen states.

| street | states | raw keys | toy keys | bucket keys | toy compression | bucket compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 4,614 | 3,296 | 232 | 1,487 | 14.21× | 2.22× |
| flop | 2,775 | 2,775 | 678 | 2,691 | 4.09× | 1.03× |
| turn | 1,784 | 1,784 | 1,080 | 1,783 | 1.65× | 1.00× |
| river | 1,077 | 1,077 | 937 | 1,077 | 1.15× | 1.00× |
| **all** | **10,250** | **8,932** | **2,927** | **7,038** | **3.05×** | **1.27×** |

Invariant: each encoder is a function of the raw information state (it maps
every raw key to exactly one abstract key): 0 violations for toy and bucket,
so neither can exceed the raw key count. Distinct `(player, action history)`
pairs — a floor for any encoder that keeps exact history — number 2,158.
Breakdowns by position, SPR bucket and betting depth are in the JSON. A
second corpus with the 50/100/200% menu gives the same picture (8,792 raw,
2,756 toy, 6,896 bucket keys).

**Fixed action lines** (2,000 deals, one history each) isolate card
abstraction:

| line | states | raw keys | bucket keys | toy keys | distinct current-street bucket labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| preflop BTN unopened | 2,000 | 1,038 | 169 | 6 | 169 |
| flop, BB first after BTN raise/call | 2,000 | 2,000 | 1,471 | 8 | 69 |
| flop, BB facing c-bet | 2,000 | 2,000 | 1,471 | 8 | 69 |
| turn, after flop check-check | 2,000 | 2,000 | 1,897 | 7 | 112 |
| river, after two check-checks | 2,000 | 2,000 | 1,992 | 9 | 80 |

**Why the bucket encoder stays so large.** Its key is
`player | preflop class / flop bucket / turn bucket / river bucket | exact history`:

* the preflop part alone has 169 classes;
* the earlier private abstraction (preflop class and each earlier street's
  bucket) must stay distinguishable for perfect recall, so it is carried
  forward;
* each street bucket also embeds a public board-texture code and an SPR
  bucket, which are retained in all later keys;
* the exact action history is retained;
* so later keys inherit the product of all earlier abstractions: the current
  flop bucket alone takes 69 values on a fixed line, but combined with 169
  preflop classes it leaves 1,471 of 2,000 flop states distinct; by the river
  1,992 of 2,000.

This does not mean perfect recall makes Hold'em impossible. It means this
particular design — independent per-street buckets, each stacked onto the
full private and public history — produces insufficient compression.

## 2. Perfect recall

Method: for every abstract key with more than one raw member in the corpus,
compare across members (up to 50 per key):

* the acting player's own earlier actions;
* the full public action history;
* the sequence of abstract keys the player held at each earlier decision
  (rebuilt by replaying history prefixes with the board dealt so far) — the
  player's remembered private and public abstract observations;
* the legal action set.

Different exact board or hole cards inside one key are the intended effect
of card abstraction and are not violations; they are recorded as
informational fields.

| encoder | abstract keys | keys with >1 member (inspected) | states in them | violations |
| --- | ---: | ---: | ---: | ---: |
| raw | 8,932 | 870 | 2,188 | 0 |
| toy | 2,927 | 692 | 8,015 | 483 (all "earlier private abstraction") |
| bucket | 7,038 | 712 | 3,924 | 0 |

The 50/100/200% corpus: bucket 720 inspected / 0 violations, toy 695 / 470.
A deliberately broken encoder that drops the history is caught (test).

**Conclusion.** No perfect-recall violations were found by the implemented
invariant audit over the tested corpus for `HoldemBucketEncoder`. This is
corpus testing, not a formal proof; structurally, the key embeds the exact
history and every earlier street's bucket, which is what the audit checks.
`ToyHoldemEncoder` is an imperfect-recall abstraction; no equilibrium
guarantee applies to anything trained on it.

## 3. MCCFR training behaviour

Setup: external-sampling MCCFR, `HoldemBucketEncoder`, 100 BB, bets
33/75/150% pot, raise cap 3; seed 0 to 5,000 iterations, seeds 1 and 2 to
1,000. One container CPU (Intel Xeon 2.1 GHz); single process each.

A *visit* is one arrival at the infoset as the non-updating player (each adds
one probability vector to `strategy_sum`); infosets reached only as the
updating player show 0. Visit totals are floats like 0.9999999999999999, so
the earlier histogram (which tested `v == 1`) silently dropped such
infosets and its percentages summed to ~93%. Counts are now verified to be
within 1e-6 of an integer and then rounded (a failure would raise).

Seed 0:

| iterations | wall clock (s) | it/s (segment) | infosets | new since previous | 0 visits | exactly 1 | 2–5 | 6–20 | >20 | max | median | mean | checkpoint | infoset memory est. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 111.5 | 1.81 | 77,872 | 77,872 | 26,483 (34.01%) | 50,788 (65.22%) | 590 (0.76%) | 11 (0.01%) | 0 | 7 | 1 | 0.672 | 29.1 MB | 34.5 MB |
| 1,000 | 527.9 | 1.94 | 355,296 | 277,424 | 121,190 (34.11%) | 229,876 (64.70%) | 3,614 (1.02%) | 616 (0.17%) | 0 | 16 | 1 | 0.689 | 132.7 MB | 157.4 MB |
| 2,000 | 1,031.7 | 2.02 | 682,750 | 327,454 | 234,041 (34.28%) | 440,326 (64.49%) | 6,703 (0.98%) | 1,528 (0.22%) | 152 (0.02%) | 28 | 1 | 0.698 | 257.5 MB | 302.6 MB |
| 5,000 | 2,531.4 | 2.03 | 1,647,628 | 964,878 | 566,954 (34.41%) | 1,058,420 (64.24%) | 18,024 (1.09%) | 3,317 (0.20%) | 913 (0.06%) | 64 | 1 | 0.710 | 621.0 MB | 730.5 MB |

Discovery vs learning ("meaningfully trained" reported at three fixed
thresholds, none privileged):

| iterations | newly discovered | revisited (≥2) | ≥5 visits | ≥10 visits | ≥20 visits | share of all visits landing in ≥5-visit infosets |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 100% | 0.77% | 0.024% | 0% | 0% | 0.2% |
| 1,000 | 78.1% | 1.19% | 0.243% | 0.070% | 0% | 2.8% |
| 2,000 | 48.0% | 1.23% | 0.320% | 0.108% | 0.029% | 4.5% |
| 5,000 | 58.6% | 1.35% | 0.307% | 0.152% | 0.060% | 6.1% |

Infoset count grows roughly linearly (≈322 new infosets per iteration
between 2,000 and 5,000) with no sign of saturation.

**Answer.** At 5,000 iterations the solver is primarily discovering new
information sets, not learning policies at recurring ones: 58.6% of
infosets first appeared after iteration 2,000, the median infoset has been
visited once, and 93.9% of all visits land in infosets visited fewer than 5
times. The current abstraction is not suitable for efficient full-game MCCFR
training.

### Canonical spots (seed 0)

Policies are average strategies; `UNVISITED` means the infoset was never
visited and has no learned policy (the uniform fallback is not shown as one).

| spot | 200 | 1,000 | 2,000 | 5,000 |
| --- | --- | --- | --- | --- |
| BTN open AA | UNVISITED | 3 visits: call .26 r33 .03 r75 .47 r150 .24 | 6: call .42 r33 .01 r75 .27 r150 .29 | 22: call .71 r33 .00 r75 .21 r150 .08 |
| BTN open AKs | UNVISITED | 1: uniform (no update yet) | 4: fold .09 call .04 r33 .23 r75 .33 r150 .22 ai .10 | 13: fold .05 call .22 r33 .32 r75 .18 r150 .17 ai .05 |
| BTN open AKo | 3: uniform (no update yet) | 10: fold .13 call .26 r33 .20 r75 .18 r150 .08 ai .14 | 19: fold .10 call .31 r33 .18 r75 .20 r150 .10 ai .11 | 42: fold .06 call .22 r33 .18 r75 .17 r150 .23 ai .13 |
| BTN open 72o | 1: fold .35 call .09 r75 .18 r150 .39 | 7: fold .29 call .15 r33 .04 r75 .03 r150 .49 | 14: fold .40 call .08 r33 .02 r75 .02 r150 .48 | 42: fold .62 call .03 r33 .02 r75 .05 r150 .29 |
| BB flop first: top pair / NFD | UNVISITED | UNVISITED | UNVISITED | UNVISITED |
| BB flop first: air | UNVISITED | UNVISITED | UNVISITED | 2 visits, still uniform (no update yet) |
| BTN after check: top pair / NFD / air | UNVISITED | UNVISITED | UNVISITED | UNVISITED |
| BB facing c-bet: set / middle pair / NFD / air | UNVISITED | UNVISITED | UNVISITED | UNVISITED |

L1 movement between consecutive checkpoints (max 2.0): AA 0.43, 0.56;
AKs 0.54, 0.54; AKo 0.18, 0.30; 72o 0.42, 0.21, 0.50. The preflop policies
are not stabilizing — they move substantially at every checkpoint, and AA
drifting toward a 71% limp at 22 visits is sampling noise, not a learned
strategy. No postflop canonical spot has a learned policy.

### Seed-to-seed disagreement (1,000 iterations, seeds 0/1/2)

Maximum possible L1 = 2.0. Unvisited infosets are excluded from every average.

| pair | canonical spots visited by both | canonical mean L1 | top-2,000 overlap | common top-2,000 mean L1 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 0 vs seed 1 | 4 | 1.245 | 43.9% | 0.618 | 0.556 |
| seed 0 vs seed 2 | 4 | 0.610 | 43.6% | 0.624 | 0.586 |
| seed 1 vs seed 2 | 4 | 1.115 | 45.0% | 0.597 | 0.565 |

Canonical spots: visited by all seeds 4 (the preflop ones; per-spot L1 e.g.
AKs 1.667 / 0.354 / 1.494), by only some 2 (BTN-after-check top pair: seed 1
only; air: seed 2 only), by none 8.

## 4. Decision-quality invariants

`tests/test_decision_invariants.py` (opponent responses pinned by a
fixed-probability model so EVs have closed forms). All pass:

1. river nuts never folds (free check and facing a bet);
2. zero-equity call costs exactly the call (EV −50.0, SE 0);
3. equity below pot odds → fold dominates call (exact formula within 4 SE);
4. equity above pot odds → call beats fold;
5. always-fold opponent → every bet size is worth exactly the pot (12.0);
6. always-call river bet → `pot + X` when winning, `−X` when losing, exactly;
7. equity < ½ with constant call responses → EV strictly decreasing in bet
   size, paired differences exactly `(2ŝ − 1)ΔX`, recommendation = check;
8. multiway pot shares not double-counted (2-way tie in 3-way pot = 0.5);
9. folded players never win (folded nut hand; EV = pot + call exactly);
10. side-pot eligibility in rollouts (short all-in nut hand wins only the
    main pot; hero EV = 80 − 50 exactly).

Split-pot regressions: (a) 3-way pot, 2-way tie, third player loses — exact
enumeration and both sampling methods give exactly 0.5 to each tied player
(also with unequal contributions, layer by layer); (b) the rules engine
still pays a real 3-chip pot 2/1 by the odd-chip rule.

Bug found and fixed: analysis code used integer pot units, so the odd-chip
rule biased split pots (2/3 vs 1/3). Pot units are now scaled by
2520 = lcm(1..9) in analysis only.

## 5. Reproducibility migration (found by CI)

The first CI run failed two Leduc pins on GitHub runners while passing
locally. Cause, reproduced exactly with `OPENBLAS_CORETYPE=Haswell`:
`strategy @ child_values` used OpenBLAS `ddot`, whose summation order
depends on the CPU kernel. A ~1e-14 difference changed Leduc CFR digests and
flipped one MCCFR sampling decision, sending seeded Leduc MCCFR down a
different trajectory (game value −0.1148 vs −0.0759). Solvers now use an
exactly rounded sum (`solvers.cfr.strategy_dot`, `math.fsum`); outputs are
identical under OpenBLAS SkylakeX/Haswell/Prescott kernels and with NumPy
AVX2/AVX-512 disabled, at no measurable speed cost. The 4 Kuhn pins were
unchanged; the 3 Leduc pins were re-recorded (old values documented in
`tests/test_reproducibility.py`). Committed research CSVs predate the change
and are unaffected beyond floating-point noise. The Phase 25 MCCFR runs
above were trained before this change on this machine's SkylakeX kernel; a
rerun elsewhere would follow a different (equally valid) sampled
trajectory.

## 6. Observer

* Synthetic: see `results/data/observer_synthetic_validation.csv` (cards
  100%, numeric 97–100% at ≥ 1280×800, weaker at small sizes) — synthetic
  renderer only.
* Real PokerNow: `tests/fixtures/pokernow/` (README, `raw/`, `annotations/`)
  defines the structure and the `pokeralpha.screenshot_annotation/v1`
  format; `python experiments/observer_validation.py --fixture-dir
  tests/fixtures/pokernow` reports per-field accuracy and MAE. Current
  output: `0 real fixtures found`, `real PokerNow accuracy: NOT MEASURED`.

## 7. CI

`.github/workflows/tests.yml`: `core` (Python 3.11 and 3.12, non-slow suite)
and `vision-ui` (observer, fixture validation, headless Streamlit). First run
(commit 34a9299): vision-ui passed; core failed on the two Leduc pins above,
which led to the migration in section 5. Status of later runs: see GitHub.

## 8. Phase 26: scalable abstractions

Script: `experiments/phase26_abstraction.py`; data:
`results/validation/abstraction_v2.json`. Same frozen corpus as section 1
(2,000 random-policy hands, 8,932 decision states, 33/75/150% menu, 100 BB)
plus the fixed lines; training runs of `holdem_mccfr_validation.py
--encoder X` with seeds 0/1/2 and checkpoints at 200/1,000/5,000
iterations (bucket rows reuse the Phase 25 runs).

### Corpus compression and recall

| encoder | keys (raw 8,932 states) | flop line keys (raw 2,000) | turn | river | recall violations / colliding keys |
| --- | --- | --- | --- | --- | --- |
| raw | — | — | — | — | 0 / 870 |
| bucket | 7,038 | 1,471 | 1,897 | 1,992 | 0 / 712 |
| transition (exact) | 6,877 | | | | 0 / 825 |
| transition_abstract | 6,630 | | | | 211 / 1,005 |
| compact_exact | 5,864 | | | | 498 / 1,142 |
| **compact** | **4,643** | **120** | **176** | **142** | **874 / 1,508** |

### Within-key coherence of `compact` (equity vs a uniform random hand)

| street | mean within-key equity std | p90 equity range | keys mixing hand categories |
| --- | --- | --- | --- |
| flop | 0.072 | 0.40 | 27% |
| turn | 0.073 | 0.42 | 32% |
| river | 0.089 | 0.50 | 51% |

Corpus-wide: 0.04% of states in keys whose equity range exceeds 0.5, 4.7% in keys
with range > 0.3, 0 keys mixing legal-action menus. Category mixing is mostly
adjacent rungs of the same strength class by design (e.g. middle pair and
weak top pair share rung 3); river mixing is highest because draws are gone.

### Training (seed 0, 5,000 iterations)

| encoder | infosets | new / iter (last segment) | >= 5 visits | >= 20 visits | canonical visited | wall clock | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bucket | 1,647,628 | 321.6 | 0.31% | 0.06% | 5 / 14 | 2,531 s | 621 MB |
| transition | 1,514,490 | 293.7 | 0.44% | 0.07% | 8 / 14 | 333 s | 515 MB |
| transition_abstract | 711,868 | 120.3 | 6.5% | 0.63% | 13 / 14 | 383 s | 235 MB |
| compact_exact | 1,212,030 | 226.8 | 1.5% | 0.15% | 13 / 14 | 319 s | 374 MB |
| **compact** | **60,747** | **4.9** | **50.8%** | **24.0%** | **14 / 14** | **438 s** | **19 MB** |

Compact: median visits 5, mean 24, 96.9% of all visits land in infosets
with >= 5 visits; 32% of infosets were first discovered in the last segment
(bucket 59%, transition 78%). Infoset counts agree across seeds (59,825-61,023).
(The bucket wall clock predates the Phase-26 feature caches and is not a
like-for-like speed comparison.)

Seed disagreement for compact (L1, max 2.0): canonical spots 0.82-1.01 at
5,000; top-2,000 visited infosets overlap 81% and their common-key mean L1
fell from ~0.64 at 1,000 to ~0.45-0.48 at 5,000. Perfect-recall candidates
overlap only 38-51% and their common-key L1 rises with training (new rare
keys enter the top set).

**Conclusion.** Only `CompactHoldemEncoder` materially improves
revisitation, and it is imperfect recall: MCCFR on it carries no standard
equilibrium guarantee and its output must be described as an abstract
strategy, not an equilibrium. The perfect-recall transition encoder fails
because the exact betting history dominates the key space; abstracting the
history is what makes revisitation possible, and that forfeits recall.

## 9. Phase 27: abstraction selection

`experiments/phase27_lock.py` scores every Phase-26 candidate against
thresholds fixed in the script: >= 25% of infosets with >= 5 visits and
>= 10x the bucket encoder; <= 50% of infosets newly discovered in the last
training segment; >= 12 of 14 canonical spots visited by all three seeds;
no legal-action mixing; <= 1% of states in keys with equity range > 0.5,
<= 10% with range > 0.3, mean within-key equity std <= 0.05.

| encoder | revisitation | not mostly discovery | canonical coverage | legal | coherence | selected |
| --- | --- | --- | --- | --- | --- | --- |
| bucket | fail | fail (59% new) | fail | pass | pass | |
| transition | fail | fail (78%) | fail | pass | pass | |
| transition_abstract | fail (6.5%) | fail (68%) | pass | pass | pass | |
| compact_exact | fail (1.5%) | fail (75%) | pass | pass | pass | |
| **compact** | **pass (50.8%)** | **pass (32%)** | **pass (12/14 all seeds, 14/14 seed 0)** | pass | pass (std 0.020, 4.7% > 0.3, 0.04% > 0.5) | **yes** |

Gate 27 -> 28: primary selected; every canonical postflop spot has >= 5
visits at 5,000 iterations (7-496); training is no longer mostly discovery;
collision quality within the thresholds. The encoder is imperfect recall
(874 / 1,508 colliding keys); this is stated wherever its output appears.

## 10. Phase 28: backend performance

Profile first (`cProfile`, locked config, 200 iterations after warm-up):
30.0 s, of which action-history replay (`_tokens`, `_replay`,
`betting_context`) was about 60% and card-feature cache misses about 25%.

Changes (all pure Python, bit-for-bit equivalent): cached action-string
tokenizer; `_replay`, `legal_actions` and the abstract betting-context key
memoized per game by action history (bounded tables); a flush-only blocker
helper instead of a full blocker scan whose straight part was discarded; an
inverse-CDF sampler equal to `Generator.choice(p=...)`. After: 13.3 s.

`experiments/phase28_benchmark.py`, baseline commit 5a42e03 vs optimized,
same machine, seed 0, 300 warm-up + 600 timed iterations
(`results/validation/backend_benchmark.json`):

| metric | baseline | optimized |
| --- | --- | --- |
| iterations / s | 15.2 | **30.1** (1.99x) |
| iteration latency median / p95 | 59.7 / 140.6 ms | 28.5 / 74.7 ms |
| infoset touches / s | 5,944 | 11,810 |
| infosets after 1,000 iterations | 41,066 | 41,066 |
| table estimate | 22.1 MB (538 B / infoset) | 21.6 MB |
| max RSS | 118 MB | 194 MB (memo tables) |
| checkpoint (13.0 MB) save median / p95 | 0.095 / 0.147 s | 0.100 / 0.152 s |
| checkpoint load median / p95 | 0.166 / 0.217 s | 0.157 / 0.265 s |
| exact SHA-256 of trained tables | b81170d8... | b81170d8... (identical) |

Equivalence: `tests/test_performance_equivalence.py` pins the
pre-optimization exact digests for the compact, bucket, transition and raw
encoders and checks the sampler against `Generator.choice`; Kuhn and Leduc
pins in `tests/test_reproducibility.py` are unchanged.

Not done, deliberately: integer infoset IDs / compact arrays (the locked
config's table is tens of MB and not a bottleneck) and a compiled backend
(remaining time is spread across the evaluator, feature misses and the
traversal; the Phase-29 budget fits in about an hour per seed; a port of
rules + features + encoder would add semantic-equivalence risk for no
required gain).

## 11. Phase 29: training the locked config

Runs: `holdem_mccfr_validation.py --locked-config`, seeds 0/1/2, checkpoints
at 1k / 3k / 10k / 30k / 100k (log-spaced), ~23-27 it/s per process, about
70 minutes per seed to 100k. Analysis: `experiments/phase29_analysis.py`
-> `results/validation/holdem_training_v1.json`. Checkpoints (~25 MB each)
stay outside the repository; the exported strategy artifact
`results/strategy/holdem_v1_seed0.npz` (1.4 MB, seed 0, 100k iterations,
every visited infoset) is committed. **The abstraction is imperfect recall:
everything below is a proxy, not a convergence or exploitability result.**

### Convergence proxies (seed 0; seeds 1 and 2 within +/- 0.01 on every rate)

| iterations | infosets | new / iter | newly discovered | >= 5 visits | >= 20 visits | top-2000 movement (L1 vs prev.) | matrix spots visited / >= 20 | visit-weighted entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1k | 41,066 | 41.1 | 100% | 28.4% | 6.4% | — | 464 / 51 | 1.07 bits |
| 3k | 54,652 | 6.8 | 25% | 43.9% | 17.3% | 0.36 | 544 / 167 | 0.94 |
| 10k | 67,705 | 1.9 | 19% | 59.5% | 34.2% | 0.36 | 592 / 400 | 0.81 |
| 30k | 76,590 | 0.44 | 12% | 70.4% | 49.1% | 0.28 | 633 / 503 | 0.72 |
| 100k | 82,781 | 0.09 | 7.5% | 79.2% | 62.5% | 0.24 | 656 / 591 | 0.65 |

Discovery has nearly stopped (0.09 new infosets per iteration). The policies
of the most visited infosets still move between checkpoints (mean L1 0.24),
and the canonical-matrix spots move about 0.5 L1 per checkpoint without a
clear downward trend, so the strategy is **not settled** at 100k.

### Seed disagreement (L1, max 2.0)

| iterations | top-2000 overlap | top-2000 common-key mean L1 | matrix spots visited by both: mean L1 | both >= 20 visits: mean L1 |
| --- | --- | --- | --- | --- |
| 1k | 75% | 0.64 | 0.73 (n ~ 375) | 0.91 (n ~ 34) |
| 10k | 84% | 0.39 | 0.93 | 0.92 |
| 100k | 88-89% | 0.25 | 0.87-0.89 (n ~ 640) | 0.84 (n ~ 540) |

The heavily visited core agrees more and more across seeds; specific
canonical spots still differ a lot between seeds (mean L1 ~0.85), so any
single spot's frequencies should be read as rough tendencies, not as stable
numbers.

### Canonical matrix

666 spots (`validation/canonical_matrix.py`): 16 preflop hands x 6
situations (BTN unopened; BB vs limp, 33/75/150% raise, all-in) and, for
five flop textures (dry high, wet connected, paired, monotone, low
connected), hand categories (monster, top pair+, medium, weak made, draw,
air) x preflop lines (limped, 75% and 150% raise: different SPRs) x nodes
(BB first, BTN after check, BB facing 33/75/150%) on the flop, plus turn
and river lines. At 100k, 656 / 639 / 653 spots are visited (seeds 0/1/2);
the 10 unvisited seed-0 spots (mostly monsters facing bets in limped pots)
are reported as `UNVISITED`. Illustrative seed-0 policies: BTN 72o folds
68%; BB vs a 75% open folds 72o 62% and 3-bets/jams AA 99.6%; facing a 75%
c-bet on K72 rainbow, air folds 93% and medium pairs call 66%; on 986
two-tone after a check the BTN bets top pair+ 99.9% and checks draws 92%.
Visible oddities, reported as they are: AA limps on the BTN 89% of the
time, and preflop all-ins are frequent — consistent with an abstract game
with a three-size menu and imperfect recall, not with real-game theory.

### Strategic sanity checks (`>= 5` visits per spot used)

All 10 pass on all three seeds: AA never folds preflop (max fold 0.4-2%);
BTN folds 72o more than AA; BTN raises premiums more than trash; BB folds
trash more than premiums vs raises and vs all-ins; facing flop bets air
folds more than top pair+ (69-76% vs 2-4%); monsters fold <= 2%; air and
weak made hands fold more to 150% than to 33% bets; first to act, value
hands bet more than weak made hands; river monsters facing a bet fold
<= 0.4%. These are coarse plausibility checks, not optimality tests.

### Duplicate cross-play with seat swap (200,000 deals per match)

Each deal is played twice with the cards fixed and seats swapped; result
for A in bb/100 hands (95% CI). Inside the abstract game only; unvisited
infosets play uniform (miss rate < 0.02% for 100k strategies).

| A | B | bb/100 | 95% CI |
| --- | --- | --- | --- |
| seed 0 @100k | seed 1 @100k | +5.4 | -4.2 .. +15.1 |
| seed 1 @100k | seed 2 @100k | -1.2 | -10.9 .. +8.5 |
| seed 0 @100k | seed 2 @100k | +13.3 | +3.6 .. +23.0 |
| seed 0 @100k | seed 1 @10k | +264.0 | +252.2 .. +275.8 |
| seed 0 @100k | seed 1 @1k | +439.9 | +427.1 .. +452.7 |
| seed 0 @100k | uniform random | +498.0 | +485.6 .. +510.4 |
| seed 0 @100k | calling station | +472.1 | +461.1 .. +483.0 |
| uniform (control) | uniform | +3.9 | -10.0 .. +17.7 |

More training clearly produces a stronger abstract strategy (100k beats 10k
by ~2.6 bb/hand); independently seeded 100k strategies are close to each
other but not identical. None of this bounds exploitability.

## Exploitability

Not computed. Exact Hold'em exploitability is infeasible here, and no proxy
in this document is an exploitability bound.
