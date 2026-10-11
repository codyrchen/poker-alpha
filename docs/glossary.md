# Glossary (one sentence each)

- **Regret** — how much better an action would have done than what you played, in hindsight.
- **CFR** — an algorithm that repeatedly plays a game against itself and shifts toward actions it regrets not taking; provably approaches equilibrium in two-player zero-sum games.
- **MCCFR** — CFR that samples chance outcomes instead of enumerating them, trading exactness per iteration for speed at scale.
- **Information set (infoset)** — all game states a player cannot tell apart given what they know; strategies are defined per infoset, not per state.
- **Mixed strategy** — playing actions with deliberate probabilities; in hidden-information games randomizing is often the unexploitable choice.
- **Exploitability** — how much a perfect counter-strategy would win per hand; zero at an exact equilibrium.
- **Best response** — the maximally exploiting strategy against a fixed known opponent.
- **Abstraction** — grouping similar states so a huge game fits in memory; its error is a measurable, not ignorable, quantity.
- **Confidence gate** — PokerAlpha's per-state evidence check (visits, seed agreement, known risk) deciding accept / low confidence / reject; an evidence gate, not a probability of correctness.
- **Rollout** — a Monte Carlo EV estimate under an explicit opponent model; a different methodology from the solver, never mixed with it in comparisons.
