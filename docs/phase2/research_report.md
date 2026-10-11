# PokerAlpha Phase II — research report

*Not a peer-reviewed paper. Every number below is in a committed artifact
under `results/phase2/` and is reproducible with the command in the linked
per-topic document. Literature is cited to place results, not to claim
novelty.*

## Abstract

Phase II asked whether PokerAlpha's solver could be made measurably
smarter, faster and better understood, under controlled baselines with
independent confirmation. Six findings are positive:

* **Error-guided abstraction refinement.** On exact 52-card river
  subgames, splitting the bucket with the largest measured best-response
  gain lowers exploitability 24-31 % against a uniform abstraction with
  identical capacity, and 44-57 % against random splits. Confirmed on new
  seeds and boards and at double capacity. No new Hold'em abstraction or
  release was trained from it.
* **Native speedup.** A profile-driven change made release-config training
  1.96x faster with bit-identical output (A/B experiment, 858 → 1,678 it/s).
  The separate fresh end-to-end benchmark measures 1,625 vs 21.9 it/s warm
  training throughput, 74.2x the Python reference.
* **Safe resolving (exact single-street river subgames only).** Decision-time
  re-solving with a re-solve gadget never
  did worse than the blueprint beyond 1e-4 chips at budgets of 1 s or more,
  and closed 9-31 % of the gap to equilibrium. Unsafe resolving was 2-3x
  *more* exploitable.
* **Translation (reduced-subgame measurement; the release is unchanged).**
  Randomized pseudo-harmonic translation of off-tree bets
  is about 8x cheaper than the deterministic nearest-in-log-space rule the
  release uses, in exact subgames. Confirmed on independent seeds.
* **Red-team mechanism.** Confirmed local-sensitivity failures are
  action-abstraction boundary discontinuities, not undertrained policies.
  A pre-registered training experiment (2M → 4M, three seeds) came out NULL
  as predicted.
* **Seed disagreement is structural.** It shrinks by about 10 % per doubling
  of training, versus about 29 % expected for sampling noise.

Five ideas failed under controls and are reported as such: prioritized
compute allocation, a confidence-v3 score, longer training under imperfect
recall, policy distance as a quality proxy, and solver regret as a
refinement signal.

## 1. Questions

1. Can measured error decide where abstraction capacity goes? (Program 1)
2. Can compute be allocated to unresolved regions more efficiently? (P2)
3. How do CFR-family algorithms compare under fair normalizations? (P3)
4. Which confidence signals carry the gate? (P4, P47-49)
5. What do capacity, bet menus, recall and collisions cost exactly? (P5)
6. Does decision-time resolving help, and is it safe? (P6)
7. Where does native training time go? (P7, P21-23)
8. Are red-team failures real, and what causes them? (P10, P11, H3-H5)
9. How general are the solver stack and the agent framework? (P14-16)
10. What does the release strategy itself look like? (P8, P17, P18)

## 2. Methods and baselines (common to all studies)

* **Exact domains.** Kuhn, Leduc, reduced preflop Hold'em, 52-card river
  subgames (pot 10, stack 20, bets 50/100 % + all-in, raise cap 2), an
  auction duel and a new signaling game. Exploitability is computed by exact
  best response.
* **New vectorized river solver** (`poker_alpha/research/river_solver.py`).
  It matches the generic CFR+ solver to 1e-15 and runs about 30x faster,
  which made multi-seed, capacity-matched studies feasible.
* **Controls.** Capacity-matched uniform and random refinement; an
  information-free reweighting control for sampling; a blueprint and a
  full re-solve for resolving; equal iterations, equal nodes and equal
  wall-time normalizations for algorithms; fixed versus nested
  leave-one-game-out thresholds for the gate.
* **Discovery, then confirmation.** Every positive claim was re-run on
  independent seeds, and boards where applicable. Discovery-only results
  are labelled as such.
* **Statistics.** Bootstrap 95 % CIs on paired differences, Wilson
  intervals for rates, Bonferroni where many comparisons were ranked,
  effect sizes reported alongside every test.

## 3. Positive results

| result | effect | uncertainty / replication | doc |
|---|---|---|---|
| Error-guided refinement vs equal-capacity uniform | -24 % (confirmation), -31 % (discovery, high capacity) exploitability | CI of the gain 0.024-0.048 chips; 39/45 wins; every board individually | adaptive_abstraction.md |
| ...vs random splits | -56 % | 45/45 wins | same |
| ...along the whole capacity frontier | adaptive better at 86 % of 540 equal-infoset points | 27 instances | abstraction_studies.md |
| Native MCCFR throughput | 858 → 1,678 it/s (1.96x), identical digest | 3-5 alternating A/B reps, < 1 % spread | native_systems.md |
| Safe resolving (re-solve gadget) | closes 22-32 % of the gap (first action), 9-12 % (second) | 0/108 settings worse beyond 1e-4 at ≥ 1 s | local_resolving.md |
| Pseudo-harmonic translation vs nearest-log | 0.60 → 0.076 chips IP cost (release-like menu) | 26/27 and 27/27 instance wins, two independent runs | action_translation.md |
| Confirmed red-team clusters | 17/18 top clusters confirmed; local sensitivity 21-26x lift | independent 10k-state sample; 90 % / 97.5 % state-level confirmation | red_team_confirmation.md |
| Safe nested resolving after off-tree bets vs pseudo-harmonic | a further 8-22 % lower translation cost | 24-26/27 instance wins, two independent runs | action_translation.md |
| Native benchmark (repository script) | 1,625 vs 21.9 it/s: 74.2x (Phase I 38.1x) | 3 trials, median | native_systems.md |
| Seed ensembling (3 seeds) | -20-26 % exploitability vs one seed | 8 reps x 3 games | hypotheses.md (H1) |
| Bayesian adaptive agent | exploits archetypes 2-5x more than equilibrium; recency +0.15/hand after a regime switch | 5 seeds / 8 seeds; Bonferroni | agent_research.md |
| Discounted CFR | best at large node budgets on 4/5 games, 1.4-2.7x better than CFR+ on poker games | 1 deterministic run per game; MCCFR 10 seeds | cfr_benchmark.md |

## 4. Null results

* **Active compute allocation (P2).** Importance-corrected prioritized deal
  sampling with four signals, and an exact best-response oracle, gave no
  replicated improvement over ES-MCCFR at equal node budgets. The discovery
  run's "positives" flipped sign on 20 independent seeds and matched an
  information-free reweighting control. The variance that matters lies
  below the deal (opponent actions, later chance), which deal-level
  allocation cannot reach.
* **Training-coverage diagnosis for red-team failures (P11 W1).** Doubling
  training did not shrink failing states' sensitivity faster than
  controls. This was pre-registered.
* **Confidence v3 (P47-49).** No logistic score beat the v2 rule on false
  confidence held out. v2 also transfers better across game families.

## 5. Negative results

* **More training can make an imperfect-recall strategy worse.** In Leduc,
  with round-1 history forgotten, real-game exploitability rose from 0.48
  to 0.94 under CFR+, and from 0.49 to 0.82 under the release's own
  algorithm (ES-MCCFR, uniform averaging). DCFR does not avoid it.
* **Policy distance is not value.** A distilled Leduc strategy 0.064 TV from
  equilibrium is 3,600x more exploitable. This also blocks the tempting
  conclusion from Program 8 that the texture feature can be dropped because
  its TV effect is below seed noise.
* **Solver regret is not a refinement signal.** Splitting by cumulative
  positive regret is no better than random.
* **Unsafe resolving.** Worse than the blueprint in 21-26/27 instances.
* **Outcome-sampling MCCFR** is the worst method in every game tested.
* **Lazy artifact loading** saves about 35 % steady RSS, not the
  hypothesised 10x, because checksum validation dominates the peak.

## 6. Systems findings

* Before optimization, 69.5 % of training CPU went to river percentile
  tables (1,081 evaluations plus a sort per new board). Boards almost never
  repeat under external sampling, so caching cannot help; deduplication
  did.
* After the changes: the evaluator is about 45-50 % of CPU and the flat hash
  index about 10 %. A SIMD evaluator, a bigger board cache, `-ffast-math` and
  float32 training are NO-GO, with reasons in `native_optimization_trials.json`.
* Memory is bounded by the abstraction's key space: about 146k infosets,
  131.5 MB peak RSS at 1M iterations, about 147 bytes per infoset of table.
  8/16/32 GB would need 36M/72M/145M infosets, which is unreachable with
  this abstraction.
* Reduced games rebuilt every deal state per MCCFR sample. Caching with an
  identical RNG stream gave 3.8x on that solver.

## 7. Correctness findings

* 14.75M reachable states over three reduced-deck Hold'em configurations
  satisfy 15 invariants with zero violations. Metamorphic suit, order,
  serialization and replay checks pass. Five fault-injection tests catch
  deliberately broken engines.
* The native evaluator equals the Python reference on **all 20,358,520
  six-card and all 133,784,560 seven-card hands**, plus 200k
  suit-permutation checks.
* Python and native MCCFR are bit-identical over a 10,000-iteration shared
  random tape. 100k continuous training equals 10 x 10k with save/load in
  between.
* Bugs found and fixed: the web client accepted engine responses of any API
  version; the agent tournament reset adaptive agents every hand; the first
  safe-resolving gadget used profile counterfactual values instead of
  best-response values, which was measurably unsafe.

## 8. Limitations

* All exact positive results are on single-street subgames or small games.
  None is a full-Hold'em measurement, and the release's imperfect-recall
  multi-street abstraction may behave differently. Program 5C shows it can
  behave *worse*.
* Wall-clock comparisons ran with several processes in parallel. Node and
  iteration counts are always recorded alongside.
* The release's flop and turn abstraction error is still bounded only by the
  earlier exact subgame studies. No exact full-game exploitability exists.

## 9. Next research questions

Ranked list in `docs/phase2/hypotheses.md`. Top open items:

* a randomized off-tree matcher in the decision engine, validated
  end-to-end (H4 follow-up);
* nested resolving after off-tree bets in the decision engine (H13 worked in
  exact subgames);
* max-margin resolving (H12);
* error-guided refinement on a multi-street exact domain (H14);
* a value-based collision signal for the release confidence table (H7).
