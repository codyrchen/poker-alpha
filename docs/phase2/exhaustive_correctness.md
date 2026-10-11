# Exhaustive reduced-Hold'em correctness (Phase II, Program 9)

The real `HoldemGame` rules engine (unmodified) is enumerated exhaustively
over reduced decks with small stacks and reduced bet menus. Only the chance
distribution is replaced, by full enumeration over the reduced deck. Every
reachable state is checked against 15 invariants
(`poker_alpha/research/exhaustive.py`).

Reproduce: `python experiments/phase2_exhaustive_correctness.py` (~3 min).
Artifact: `results/phase2/exhaustive_correctness.json`.
Tests: `tests/test_exhaustive_correctness.py`.

| config | deck | states | decision | terminal | raw infosets | compact infosets | max depth | violations |
|---|---|---|---|---|---|---|---|---|
| stack 4, pot bet, cap 1 | 9 (AKQ x 3 suits) | 2,484,217 | 896,616 | 1,107,540 | 582,336 | 534 | 14 | **0** |
| stack 6, 50%/100%, cap 2, NLHE min-raise | 9 | 11,755,801 | 4,168,584 | 5,649,588 | 2,921,184 | 1,260 | 15 | **0** |
| stack 3, pot bet, cap 1 | 9 (AKQJ x 2 + Ad) | 514,081 | 154,224 | 213,948 | 85,824 | 476 | 13 | **0** |

Total: 14.75 million reachable states, zero invariant violations. Every DFS
path terminated within the depth bound, so the reachable graph has no
cycles or deadlocks in these configurations.

## Invariants (every state)

Chip conservation (cached contributions equal the replayed history);
0 <= contribution <= stack; pot equals the sum of contributions; board size
matches the street; no duplicate cards; current player is 0 or 1; legal
actions non-empty, check/call always present, fold present iff facing chips;
no raise offered against an all-in opponent or without chips beyond the
call; a matched all-in closes betting; sized bets strictly below the stack;
folded states are terminal; showdown only on the river; finite utility
bounded by the contributions; depth bound.

## Metamorphic checks (3,000 sampled states per config, all passed)

Suit permutation (three permutations: legal actions, terminal utility and
the compact encoder key are invariant); hole-card order; JSON
serialization round trip; replay from the root reproduces streets,
contributions, fold and all-in flags.

One bug in the *test harness itself* was found and fixed during this work:
the first suit-permutation helper assumed `rank*4+suit` card codes, while
the engine uses `suit*13+rank`. The metamorphic test failed (utility
flipped sign) until that was corrected. The engine was not at fault.

## Fault injection (tests must catch deliberate breakage)

Each fault is injected by monkeypatching inside the test only:

| fault | caught by |
|---|---|
| contribution cache drifts by 0.25 after a call | chip-conservation invariant |
| duplicate card dealt | duplicate-card invariant |
| fold no longer terminal | folded-state invariant |
| shove offered to a player facing an all-in | all-in raise invariant |
| NLHE minimum raise disabled | minimum-bet sizing check |

Artifact SHA mismatch, strategy/config mismatch and truncated or corrupted
artifacts are already covered by `tests/test_artifact_integrity.py`.

## Size of the unreduced problem

The configurations above are what fits in minutes in Python. A 52-card deck
makes even the preflop chance node 1.6 million deals, so full enumeration
remains infeasible; this study covers rule logic, not card coverage.
