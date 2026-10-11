# Agent research with the generic agent interface (Phase II, Program 15)

Artifact: `results/phase2/agents.json`. Reproduce:
`python experiments/phase2_agents.py --workers 7` (about 1 min). Game:
Kuhn, which has an exact equilibrium and an exact best response.

## A framework bug found first

`play_match` reset both agents **before every hand**, although the
`Agent.reset` contract says "before each match". No adaptive agent could
learn across hands; stateless agents hid the bug. Agents are now reset
once per match. They may implement `observe_episode(seat, public_actions,
own_payoff)`, which receives only public information: the opponent's
private cards are never passed. The existing tournament tests still pass.

## Agents

random, two scripted archetypes (tight: bet/call only with K; aggressive:
always bet/call), the CFR+ equilibrium, two static exact best responses
(to tight, to aggressive), and two new adaptive agents
(`poker_alpha/agents/adaptive.py`). Both adaptive agents keep an exact
Bayesian posterior over the four candidate strategies, using public actions
marginalised over the hidden card, and play an exact best response to the
posterior. The response is computed through the realization-equivalent
behaviour strategy of the type mixture (Kuhn's theorem). `bayes_recency`
discounts old evidence (decay 0.97 per hand).

## Payoff matrix (row agent, chips/hand, 5 seeds x 2,000 seat-swapped hand pairs)

| | rand | tight | aggr | cfr | br_t | br_a | bayes | bayes_rec |
|---|---|---|---|---|---|---|---|---|
| **cfr_eq** | +0.155 | -0.004 | +0.108 | 0 | +0.192 | +0.049 | -0.006 | -0.003 |
| **bayes** | +0.459 | +0.254 | +0.336 | +0.006 | +0.984 | +0.159 | 0 | +0.084 |
| **bayes_recency** | +0.453 | +0.243 | +0.335 | +0.003 | +1.103 | -0.171 | -0.084 | 0 |

The full 8x8 matrix with standard errors is in the artifact.

* The equilibrium is unbeaten beyond noise. The Bayes agent's +0.006
  against it (z = 2.05) does not survive Bonferroni correction over 28
  pairings (|z| > 3.12 required). Theory bounds the true edge by the
  equilibrium's exact exploitability, 3.9e-5. Because the empirical
  meta-game Nash was computed on point estimates, it put its support on
  `bayes`. That is a noise artifact, documented in the artifact.
* The Bayes agent exploits every archetype 2-5x harder than the
  equilibrium does: +0.25 vs tight where the equilibrium gets about 0, and
  +0.34 vs aggressive where it gets +0.11.
* **Non-transitivity is real among the non-equilibrium agents.** Three
  3-cycles survive Bonferroni correction, for example tight beats
  br_vs_aggressive, which beats br_vs_tight, which beats tight.

## Posterior calibration (well-specified; 30 seeds)

Mean posterior mass on the true archetype after N hands:

| true | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| aggressive | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 |
| tight | 0.63 | 0.90 | 0.99 | 1.00 | 1.00 |
| random | 0.52 | 0.66 | 0.79 | 0.92 | 0.98 |
| equilibrium | 0.44 | 0.63 | 0.78 | 0.91 | 1.00 |

Mixed strategies are identified more slowly than deterministic archetypes.
These numbers assume the opponent is one of the candidates.
Misspecification is not modelled.

## Regime shift (tight to aggressive at hand 1,500 of 3,000; 8 seeds)

| agent | before switch | after switch | first 200 after |
|---|---|---|---|
| cfr_eq | -0.01 ± 0.02 | +0.11 ± 0.02 | +0.13 ± 0.02 |
| bayes (stationary) | +0.26 ± 0.01 | +0.18 ± 0.01 | +0.14 ± 0.04 |
| bayes_recency | +0.26 ± 0.01 | **+0.33 ± 0.01** | +0.19 ± 0.04 |
| br_vs_tight (static) | +0.24 ± 0.01 | **-1.16** | -1.18 |

Forgetting costs nothing before the switch and earns about 0.15
chips/hand more after it. The stationary posterior needs about 1,500
hands of contrary evidence to overturn its belief. The static exploiter
collapses. This reproduces, through the generic interface, the Phase I
stationary-vs-recency result.

## Fingerprints

The behavioural summary against a random opponent (open-bet and call
frequencies by card) is in the artifact. The equilibrium opens with J 22 % and with K 78 % of the
time; these pool both seats' opening decisions, so they are not the
textbook first-player parameters. The adaptive agents become near-pure
exploiters.
