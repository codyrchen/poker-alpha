# General game core stress test (Phase II, Program 14)

New game: `poker_alpha/games/signaling.py`, a zero-sum Bayesian signaling
game. Chance draws the sender's private type. The sender sends a message,
optionally costly and type-dependent. The receiver sees only the message and
acts. Unlike Kuhn, Leduc and the auction, private information is one-sided
and the strategic content is entirely in what a message reveals.

**No solver-core modification was required.** CFR, CFR+, DCFR, LCFR,
ES-MCCFR and OS-MCCFR ran unchanged. Exact best response and exploitability
ran unchanged.

## Theory oracles (all hold)

| variant | receiver value (CFR+ 20k) | no-information value | theory |
|---|---|---|---|
| cheap talk, 3 types x 3 messages | 0.5000 | 0.5000 | zero-sum cheap talk cannot help: values must be equal |
| costly, but message m0 free for all types | 0.5000 | 0.5000 | sender can pool on the free message, so value is unchanged |
| beer-quiche-like, no universally free message | **0.86** | 0.80 | costs make messages informative; the receiver gains |

The posterior over types given each message is in
`results/phase2/general_games.json`.

## Final exploitability (chips), signaling variants

| variant | CFR 3k | CFR+ 3k | DCFR 3k | LCFR 3k | ES-MCCFR 200k | OS-MCCFR 200k |
|---|---|---|---|---|---|---|
| cheap talk | 6e-5 | 0 | 0 | 0 | 3.3e-4 | 6.0e-3 |
| costly (free m0) | 4.4e-4 | 0 | 0 | 0 | 8e-5 | 2.6e-2 |
| beer-quiche-like | 4.0e-3 | 7e-5 | 2e-5 | 2.6e-4 | 2.2e-4 | 4.9e-3 |

## Complexity inspector

`poker_alpha/games/inspect.py` walks any game with enumerable chance. It
counts nodes, chance, decision and terminal nodes, infosets per player,
max depth, branching and sequence-form size. It also checks solver
contracts: chance sums to 1, infosets have consistent action sets, keys are
unique across players and utilities are finite. Every game passes.

| game | nodes | decision | infosets | max depth |
|---|---|---|---|---|
| Kuhn | 55 | 24 | 12 | 4 |
| Leduc | 9,451 | 3,780 | 288 | 10 |
| Auction 5x5 | 776 | 150 | 10 | 3 |
| Reduced preflop (AKQJ x 2 suits, 8 bb) | 9,241 | 3,360 | 224 | 5 |
| River subgame 6x6 | 1,327 | 476 | 84 | 5 |
| Signaling 3x3 | 40 | 12 | 6 | 3 |

Reproduce: `python experiments/phase2_general_games.py`. Tests:
`tests/test_signaling_and_inspect.py`.
