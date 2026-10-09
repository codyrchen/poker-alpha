# Flop abstraction error (Phase 56) and remediation decision (Phase 58)

`python experiments/flop_turn_abstraction.py flop --iters 300 --range 6 --turns 4 --rivers 3`
-> `results/validation/flop_abstraction_v1.json` (8,560 s, 7 flops).
Same method as `docs/turn_abstraction.md`: exact subgames of the real
`HoldemGame` rules from the flop (preflop raise-call, pot 5 BB, 100 BB
stacks), 6-combo ranges per player (polarized vs medium), a fixed sample of
4 turn and 3 river cards, one bet size (75%) + all-in, raise cap 1, CFR+
300 iterations; abstract strategies lifted to raw information sets.

## Results (BB per hand; pot 5 BB)

Exploitability in the raw subgame (EV error in brackets):

| flop | raw | bucket (perfect recall) | compact (v1 release) | compact + river pct20 (**v2 release**) |
| --- | --- | --- | --- | --- |
| A72 rainbow | 0.023 | 0.73 (0.002) | 3.52 (0.004) | **5.08** (0.005) |
| K83 two-tone | 0.020 | 0.33 (0.109) | 2.18 (0.031) | **2.72** (0.029) |
| 987 two-tone | 0.027 | 1.08 (0.128) | 2.72 (0.063) | **2.80** (0.046) |
| 762 rainbow | 0.022 | 1.08 (0.001) | 1.87 (0.002) | **2.20** (0.000) |
| paired KK5 | 0.013 | 0.65 (0.001) | 6.19 (0.004) | **4.89** (0.011) |
| monotone J84 | 0.024 | 0.41 (0.031) | 4.15 (0.128) | **3.71** (0.131) |
| connected QJT | 0.031 | 1.97 (0.415) | 3.57 (0.011) | **4.20** (0.017) |

Abstract information sets: raw 3,672-4,158; bucket 2,313-3,132; compact
444-726; v2 675-951. Reach-weighted collision severity of v2: 0.02-0.15.

## Reading

* As on the turn (Phase 57), the compact encoders' lifted strategies are
  **2-6 BB/hand exploitable** in these subgames, ~100x the raw reference;
  perfect-recall buckets sit in between (0.3-2.0). v2's river buckets do
  not change flop behaviour materially (within +-1.5 BB of v1).
* The value of the profile is far less affected: v2 EV error <= 0.13 BB
  (<= 3% of the pot) on every flop.
* Same caveats as the turn study: tiny ranges, sampled runouts, one bet
  size, all-in available at 100 BB, one board per texture. These numbers
  show that the merges are strategically real, not how exploitable the
  full-game strategy is.

## Phase 58 remediation decision

Material? **Yes** for strategy quality (worst-case exploitability), not for
values (EV error small). Options considered:

| option | decision |
| --- | --- |
| change the v2 encoder or the v2 artifact | **no** — v2 is never mutated |
| new v3 encoder (e.g. flop / turn strength-percentile buckets, like v2 did for the river) and retrain 3 seeds | **not now**: a new config signature, ~3 CPU-hours per seed per 100k iterations plus re-running these studies; recorded as the main solver follow-up |
| tighten the confidence gate on flop / turn | **done**: every flop and turn lookup with a compact-encoder strategy gets the coded reason `STREET_ABSTRACTION_ERROR` and is capped at LOW confidence (ACCEPT -> LOW; REJECT unchanged). Preflop and river lookups are unaffected. |

Effect on the release strategy: the flop (17.9%) and turn (20.0%)
visit-weighted ACCEPT shares from Phase 64 become LOW confidence; ACCEPT
remains possible only preflop (8.7%) and on the river (21.0%). Decisions in
those spots still show the solver frequencies, labelled low confidence, with
rollout EVs alongside.
