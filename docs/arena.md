# PokerAlpha Arena (/arena)

A heads-up Kuhn sandbox owned entirely by PokerAlpha — NOT an interface
to any third-party poker client. The visitor plays against the committed
CFR policy (exploitability 0.0015 chips/hand) in the browser tab; all
state is local; chips are game units (no currency, no wagering UI).

Session: 10 / 25 / 50 hands, seats alternating. During play the
opponent's card and mix are hidden (revealing them would invalidate the
game); everything is revealed afterwards.

## X-ray report
- verdict with sample-size honesty (< 10 hands: "far too few to say
  anything"; otherwise mean chips/hand with a 95% CI);
- EXACT exploitability of the visitor's empirical strategy (their
  observed frequencies, unobserved infosets uniform — stated), computed
  by the same best-response code validated against Python;
- every decision side-by-side with the trained mix.

## Strategy fingerprint (docs/strategy_fingerprint.md)
Observed tendencies in THIS sample — aggression, fold-to-bet, Jack bluff
rate (observable: the player's own card is known), King value-bet rate,
policy surprise (−log2 P under the trained mix — unusualness, never
"mistake"). Each dimension shows n and a CI; dimensions under n=5 say
"insufficient data". A share card is rendered client-side to PNG; no
uploads, no accounts, no ratings.

Integrity is tested in `arena.test.ts`: legal transitions, rule-table
settlement, deterministic seed replay, out-of-turn rejection,
small-sample honesty, always-bet measured as exploitable.
