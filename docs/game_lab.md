# Hidden-Information Game Lab (/games)

The solver stack runs on the generic `poker_alpha.games.base.Game`
interface. Games included and verified:

| game | infosets | exact exploitability after CFR |
|---|---|---|
| Kuhn poker | 12 | ~0.0015 (10k iters) |
| Leduc poker | 288 | ~0.013 (1.5k iters) |
| Sealed-bid auction duel | 6 | 3e-05 (5k iters) |

The auction duel is deliberately a ZERO-SUM form (payoff = own surplus −
opponent surplus; ties award nobody) because the solvers require
zero-sum games — documented as NOT the classic general-sum first-price
auction. Its learned equilibrium shades bids below value with
monotone-in-value bidding; symmetry, monotonicity and the
hidden-information property (player 1's infoset excludes player 0's
bid) are tested.

Defining a new game = subclassing `Game` (~60 lines for the auction);
every solver, exact evaluator and tournament tool then works unchanged.
The complexity inspector (states / infosets / terminals / chance nodes)
is enumerated per game by the data build.

Not implemented (honest no-gos): a browser game-definition DSL (safe
schema design deserves more than a night; nothing ships half-secure),
negotiation/repeated games (the Game interface is single-shot
extensive-form).
