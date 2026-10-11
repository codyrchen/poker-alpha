# Strategy Fingerprint

"Observed tendencies in this sample" — never personality, never a skill
score, never a rating.

Rules enforced in code (`apps/web/src/lib/arena.ts`, tested):
- every metric carries its observation count; below n=5 the value is
  null and displayed as "insufficient data";
- rates carry binomial standard errors; the session verdict says "within
  noise" when the 95% CI covers zero;
- bluff frequency is only reported because Kuhn makes it OBSERVABLE
  (the player's own card is known); nothing is inferred from unseen
  cards;
- policy surprise (−log2 P(action) under the trained mix) is labeled
  unusualness, explicitly not regret or EV loss;
- exploitability of the empirical strategy requires ≥10 hands and ≥6
  observed infosets, and states that unobserved infosets default to
  uniform.

The share card is generated client-side (canvas → PNG download) from
summarized numbers only. Suggested share text makes the sample explicit
("after 25 Arena hands").
