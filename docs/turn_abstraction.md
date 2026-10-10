# Turn abstraction error (Phase 57)

`python experiments/flop_turn_abstraction.py turn --iters 300 --range 6 --rivers 3`
-> `results/validation/turn_abstraction_v1.json` (740 s).

## Method

Exact subgames of the real `HoldemGame` rules starting at the turn (fixed
preflop raise-call and flop check-check, pot 5 BB, 100 BB stacks), hole
cards from explicit 6-combo ranges per player (a polarized range vs a
medium-strength range), the river from a fixed sample of 3 cards, one bet
size (75% pot) + all-in, raise cap 1. Raw and abstract keys play identical
trees, so the comparison is exact **for this small game**: each encoder's
abstract game is solved with CFR+ (300 iterations), the abstract strategy is
lifted back to raw information sets and evaluated there.

* exploitability: average best-response gain per hand (BB) in the raw subgame
* EV error: |value of the lifted strategy profile - raw equilibrium value|
* weighted L1: reach-weighted L1 distance to the raw equilibrium strategy
* collision severity: reach-weighted strategic dispersion of merged keys

## Results (BB; pot 5 BB)

| turn board | raw | bucket (perfect recall) | compact (v1 release) | compact + river pct20 (**v2 release**) | v2 EV error |
| --- | --- | --- | --- | --- | --- |
| blank (Ks7d2c 3h) | 0.008 | 0.025 | 9.60 | **8.71** | 0.001 |
| overcard | 0.009 | = raw | 2.36 | **2.99** | 0.134 |
| paired turn | 0.010 | 0.010 | 6.46 | **7.17** | 0.005 |
| flush completes | 0.013 | 0.215 | 1.69 | **1.87** | 0.168 |
| straight completes | 0.018 | 0.132 | 1.89 | **1.42** | 0.073 |
| four-straight board | 0.026 | 0.675 | 6.04 | **5.65** | **0.656** |
| four-flush board | 0.011 | 1.316 | 5.87 | **5.13** | 0.029 |

(columns 2-5: exploitability in the raw subgame; "= raw": identical
information partition, so the identical solution.)

## Reading

* The compact encoders (v1 and the v2 release) merge strategically different
  turn hands: in these subgames their lifted strategies are **1.4-9.6 BB per
  hand exploitable**, versus ~0.01 BB for raw information sets and 0.01-1.3
  BB for perfect-recall buckets. The v2 river percentile buckets do not help
  on the turn (they only change river keys): v2 is within +-1 BB of v1.
* The **value** of the profile is much less affected: EV error <= 0.17 BB in
  six of seven boards; the exception is the four-straight board (0.66 BB,
  13% of the pot), where the abstraction cannot tell made straights from
  one-card draws.
* Caveats that make these numbers an upper-end illustration rather than a
  Hold'em estimate: 6-combo ranges (a best responder knows each merged
  key holds only a few specific hands), 3 river cards, one bet size, 100 BB
  behind with an all-in option (large punishments for wrong all-in
  frequencies), 300 CFR+ iterations, one board per texture.

Consequence and remediation: see `docs/flop_abstraction.md` (Phase 58
decision, made together with the flop results).
