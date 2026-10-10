# Preflop disagreement forensics (final-trust project, Phase 11-12)

Data: `results/validation/preflop_disagreement_forensics.json` — all 169
canonical classes × 6 situations, native seeds 0-2 at 200k / 300k / 1M,
enriched with the EV studies where measured. Diagnoses at 1M:

| diagnosis | count | meaning |
|---|---|---|
| STABLE | 210 | seed L1 < 0.3 |
| NEAR_EQUIVALENT | 199 | high L1, measured EV consequence below noise/pot threshold |
| CONSEQUENTIAL | **7** | high L1 AND significant cross-seed regret |
| SAMPLING_LIMITED | 85 | high L1 still falling strongly with training |
| PERSISTENT_MIXING | 509 | high L1, flat, EV unmeasured for this class/situation |
| UNDERVISITED | 4 | below the 20-visit floor |

By situation:

* **BTN unopened** (EV measured for all 169): 159 near-equivalent,
  2 consequential (A5s, 93s), 8 stable. The famous "preflop first-action
  noise" is ~94% harmless mixing between near-equal open sizes/limps.
* **BTN vs jam**: 137/169 STABLE — binary call/fold decisions converge.
* **BB vs opens / BTN vs 3-bet**: mostly PERSISTENT_MIXING. EV was measured
  only for the 12 canonical classes there; among those measured, the split
  was overwhelmingly near-equivalent (Phase 8), so the unmeasured majority
  is *likely* similar — but that is extrapolation, stated as such.

## Answers to the Phase-11 questions

| hypothesis | verdict |
|---|---|
| sampling noise | PARTIAL — 85 states still improving with training; EV regret shrinks with iterations in exact games too |
| multiple near-equivalent actions | **PRIMARY CAUSE** — measured directly (Phase 8/12); mixing sits where action margins are ~0.3 bb |
| imperfect recall downstream | NOT REQUIRED for the phenomenon (exact perfect-recall games reproduce it); it does add real postflop error (see abstraction summary) |
| action abstraction | CONTRIBUTING — three near-equal preflop raise sizes create value-flat menus (margin data); richer/poorer menus would change mixing, not value much (exact-game margins) |
| opponent policy differences | REAL in jam/3-bet spots — the 7 consequential states are exactly where call-EV depends on the opponent's still-differing range |
| rare reach / payoff variance | minor (4 undervisited states) |
| encoder collision | not implicated preflop (169-class part is lossless) |
| bug | no evidence — exact-game validation direct and bitwise parity both green |

## Bottom line

Preflop "noise" is mostly **harmless**: near-equivalent mixing that v1's
L1-based signals systematically over-punish. The dangerous residue is a
small, identifiable cluster (jam/3-bet defense with medium pairs and
suited-ace boundary hands) where range uncertainty is EV uncertainty —
these stay gated by seed disagreement under confidence v2, and keep
shrinking with training (24 → 8 significant states from 200k → 1M).
