# Abstraction error — authoritative summary (final-trust project, Phase 15)

One place for what is actually measured about the release encoder
(`CompactHoldemEncoder` + 20 river percentile buckets, imperfect recall).
Machine-readable: `results/validation/abstraction_error_summary.json`.
This supersedes any older wording elsewhere (in particular the stale
"flop/turn error not measured exactly" row that used to be in
`docs/release_status.md` — both streets ARE measured).

| street | status | release-encoder error (exact subgames) | consequence |
|---|---|---|---|
| preflop | card mapping lossless (169 classes); betting context abstract | n/a — instability is MCCFR mixing, not card abstraction (`docs/preflop_disagreement.md`) | first-action keys largely gated by seed disagreement |
| flop | **measured exactly**, 7 reduced subgames (fixed runout samples, reduced menu, pot 5 bb) | exploitability median 3.7 bb, max 5.1 bb; EV error ≤ 0.13 bb | street capped at LOW confidence |
| turn | **measured exactly**, 7 reduced subgames | exploitability median 5.1 bb, max 8.7 bb; EV error ≤ 0.66 bb | street capped at LOW confidence |
| river | **measured exactly**, six 52-card subgames | pct20: exploitability median 0.42 bb, max 0.57 bb (compact v1 was 3.5–9.6 bb: 4–16x worse) | no street cap |

Raw-key references solve the same subgames to ≤ 0.03 bb — the lift is
abstraction, not solver error.

## Full game

* Exact full-game exploitability: **not computed** — infeasible
  (`results/validation/best_response_feasibility.json`).
* New in this project: a **restricted-LBR lower bound** (validated on exact
  games, Spearman 0.986 vs true exploitability): the 200k/300k/1M
  strategies are all ≥ ~**100 bb/100** exploitable by a range-tracking
  best responder restricted to the abstract menu (`lbr_holdem.json`;
  candidates not separable at 10k hands). This is an abstraction-level
  floor: per-street errors are bounded but compound, and imperfect recall
  denies any equilibrium guarantee.

## Separation of risk dimensions (Phase 16)

* **Training uncertainty** (seed disagreement, visits, recent movement,
  value uncertainty) — improves with iterations; measured to improve
  through 1M–2M.
* **Abstraction risk** (this table + collision dispersion) — does NOT
  improve with training. Street caps and the collision signal survive into
  confidence v2 unchanged; extremely stable training must never erase them.
