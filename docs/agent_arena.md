# Agent Arena (poker_alpha/agents/)

A generic agent protocol for extensive-form hidden-information games:
`choose_action(observation, legal_actions, rng)` / `reset()` /
`metadata()` — agents see only their information-set key, so none can
condition on hidden state.

Adapters: policy table (CFR/MCCFR output, uniform fallback counted),
uniform random (mandatory baseline), scripted Kuhn archetypes with
fully documented rules, and a both-seat exact best responder (oracle /
diagnostic, honestly labeled "not a fair player").

Tournaments (`round_robin`): seat-swapped pairs share chance seeds (deal
luck pairs out in root-chance games), fixed seeds, normal-approximation
95% CIs, metadata for every agent. Correctness cross-check: the oracle's
measured edge over CFR (+0.0009/hand) matches CFR's independently
computed exploitability (0.0016) — two computations of the same
quantity agreeing.

Illegal actions raise immediately; the harness never coerces. Results on
the website are the committed `tournament.json` (4,000 paired hands per
matchup, seed 7).
