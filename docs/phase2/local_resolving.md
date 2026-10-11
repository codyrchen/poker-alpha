# Decision-time local resolving (Phase II, Program 6)

RESEARCH / SANDBOX ONLY. No poker-client connection.

**Question.** Can PokerAlpha improve a decision by re-solving the subtree at
decision time, at full hand resolution and under a time budget, instead of
relying on a coarse precomputed blueprint? Is doing so safe?

**Answers (27 instances x 2 blueprint sizes x 2 resolver positions).**

1. **Unsafe resolving is dangerous.** It assumes the opponent arrives with
   their blueprint range, and makes the resolver 2-3x *more* exploitable
   than the blueprint when resolving after the first action (BR value 3.3
   vs 1.3 chips at 1 s, K = 3). It is worse than the blueprint in 21-26 of
   27 instances, at every budget. It is better at exploiting an opponent
   who actually plays the blueprint, which is the textbook tradeoff, here
   measured exactly.
2. **Safe resolving with a re-solve gadget never made things worse beyond
   1e-4 chips at budgets >= 1 s (0 of 108 instance-settings).** It recovered
   22-32 % of the blueprint's gap to equilibrium when resolving after the
   first action, and 9-12 % for the second-action subtrees. Improvement
   saturates by about 100 ms. The gadget fixes the resolver's range at its
   blueprint reach and keeps every opponent hand at or below its blueprint
   best-response value, so most of the abstraction error, which lives in the
   ranges, cannot be repaired locally.
3. **In this tiny domain, re-solving the whole game from scratch is better
   still.** About 100 ms reaches 96-98 % of equilibrium. That only means the
   domain is small. It is the reference that shows how much error remains
   for a local method to target.

## Method

`poker_alpha/research/resolving.py`. A blueprint is K = 3 or 6 strength
cells per player, CFR+ 400 iterations, on the Program 1 river subgames:
pot 10, stack 20, bets 50 %/100 %/all-in, raise cap 2, 48 combos per
player, 9 boards x 3 seeds. The resolver re-solves each subtree at full hand
resolution with CFR+ until `solve_for_ms(budget)` expires. The budget is
split equally across the subtrees.

* **unsafe**: both ranges at the subtree root are blueprint reaches.
* **safe**: the re-solve gadget of Burch, Johanson & Bowling (AAAI 2014).
  Each opponent hand chooses to terminate, receiving its counterfactual
  best-response value against the blueprint, or to follow into the subtree.
  At the gadget optimum no opponent hand gains, which is the safety
  argument. It holds at convergence. At finite budgets it was measured, not
  assumed.

Metric: exact value of the opponent's best response against the resolver's
strategy (worst case, lower is better). Also reported: the resolver's EV
against a blueprint-playing opponent. The two are never mixed. A rollout
baseline was **not** compared, because its decision quality has no exact
best-response metric here.

## Results (mean over 27 instances; BR value in chips, lower is better)

| setting | blueprint | unsafe 1 s | safe 100 ms | safe 1 s | full re-solve 100 ms | equilibrium |
|---|---|---|---|---|---|---|
| K=3, resolve after first action (IP) | 1.315 | 3.338 | 1.191 | 1.193 | 0.711 | 0.704 |
| K=3, resolve OOP second action | -0.174 | -0.086 | -0.295 | -0.294 | -0.691 | -0.703 |
| K=6, IP | 0.928 | 2.657 | 0.856 | 0.859 | 0.711 | 0.704 |
| K=6, OOP | -0.549 | -0.306 | -0.563 | -0.563 | -0.692 | -0.703 |

Gap closed by safe resolving at 1 s: 22 % (K3 IP), 12 % (K3 OOP),
31 % (K6 IP), 9 % (K6 OOP).

## A bug the measurement caught

The first version used the opponent's blueprint *profile* counterfactual
values as terminate payoffs. Safety requires *best-response* values. With
profile values the "safe" resolver was measurably worse than the blueprint
in 3 of 27 instances even at 5 s. The corrected gadget is pinned by
`tests/test_resolving.py`, which asserts it is never worse and that its
terminate values upper-bound the profile values. The superseded result is
kept in `results/phase2/local_resolving_profile_cfv_gadget.json` and
labelled as such.

## Limitations

Single-street subtrees with no chance below the root. One blueprint
family. Budgets are wall-clock with 6 instances in parallel, and iteration
counts are recorded. Max-margin or reach-maxmargin resolving (Moravčík et
al. 2016; Brown & Sandholm 2017), which can improve on the gadget, was
not implemented.

Reproduce: `python experiments/phase2_local_resolving.py --workers 6`
(about 6 min).
