# Study Mode

`streamlit run poker_alpha/ui/app.py` → sidebar **Mode → Study**. Four
tabs, all driven by `poker_alpha/analysis/` through the same engine as
Play Mode.

## Session Review

Input: a `pokeralpha.hand/v1` file (upload) or the bundled demo session
(`poker_alpha/data/demo_session.json`, 8 HU hands that really exercise
accepted / low-confidence / rejected gate states). Every hero decision is
replayed and re-analysed by the real engine; nothing is scored by
UI-side heuristics.

### Semantic boundaries (the point of the whole feature)

* **ESTIMATED DECISION REGRET** exists only where
  `poker_alpha.analysis.regret_eligibility` says the comparison is valid:
  the EVs are the actual decision basis (rollout method), the real action
  was itself EVALUATED (an EXACT mapping — for off-menu bet/raise sizes
  the review injects the hero's exact size as an extra rollout candidate
  on the same shared samples; a nearest-size translation yields
  `ACTUAL_SIZE_NOT_EVALUATED` instead of a number), both EVs exist and
  share one methodology. Uncertainty is the PAIRED-difference SE from the
  common-random-number rollout — never an independent-SE combination.
  Everything else is **"EV regret unavailable"** with a reason — solver
  frequencies are never read as EVs.
* **Strategy alignment** classifies the action's frequency in a solver
  mix with neutral language and documented thresholds
  (`COMMON >= 25%`, `MIXED 5–25%`, `RARE < 5%`, plus
  NOT_IN_MIX / UNCERTAIN / NOT_SOLVER). `-log2(p)` is reported as
  **policy surprise**, never as regret.
* **Review categories** (documented thresholds in
  `poker_alpha/analysis/review.py`):
  * HIGH-PRIORITY EV REVIEW — eligible regret ≥ 0.5 bb and ≥ 2 SE;
  * STRATEGY REVIEW — solver ACCEPTED, action RARE or absent;
  * UNCERTAIN — solver LOW CONFIDENCE / REJECTED;
  * UNSCORED — refusal or unmappable action;
  * OK — the rest.
* **Patterns** need ≥ 3 occurrences ("Potential pattern") or ≥ 5
  ("Repeated deviation"); below that, nothing is claimed.
* The **session confidence report** keeps observer recognition and solver
  trust separate — they are never combined into one number.

Selecting a row re-renders that decision with the Play Mode renderers.
Exports (JSON / CSV / Markdown) preserve provenance, confidence, regret
eligibility and reason codes, and are written locally by the browser.

## Hand Replay

Step through every hero decision of a session (bundled, uploaded
`pokeralpha.hand/v1` JSON, or the `pokeralpha-lines/v1` text format via
the importer framework in `poker_alpha/history/importers/`). Each
decision shows the full Play rendering plus two drill-downs:

* **Modelled opponent range** — 13x13 matrices of the prior and each
  street's posterior, computed by the same `infer_opponent_range` the
  engine uses, under a selectable behavior archetype. Labeled MODELLED
  OPPONENT RANGE: a belief model, never ground truth.
* **Decision path** — the abstract infoset key, training visits, action
  menu, gate verdict/reasons/signals and size-translation notes — only
  what the engine actually recorded.

## Model Sensitivity

The same state analysed under all four behavior archetypes; verdicts
ROBUST / SENSITIVE / HIGHLY SENSITIVE follow a documented rule
(top-action agreement + total-variation distance, method changes cap the
verdict at SENSITIVE). No archetype is claimed correct; EVs are never
compared across methods.

## Strategy Explorer

A 169-hand matrix for the heads-up preflop nodes the release abstraction
represents (BTN first action, BB vs 2.5x open, BTN vs 3-bet). Every cell
is a live `SolverStrategyProvider.lookup` against the promoted artifact.
Cells the gate rejects are dimmed and show **no numbers**. Below the
matrix, any pasted HU state is inspected the same way, labeled
ACCEPTED / LOW CONFIDENCE / REJECTED / UNSUPPORTED.

## What-if + Decision Stability

One variable at a time (`villain_bet`, `pot`, `effective_stack`,
`hero_cards`, `board_card`); the variant state is fully re-validated
(duplicate cards, impossible stacks, illegal transitions are rejected
with the reason) and re-analysed. Variants are compared by distribution,
gate state, method and confidence — never by EV across methods. The same
sweep reports **decision stability**: where the highest-frequency action,
the gate state or the method changes across the swept values
(descriptions of pipeline output, not causal inference).

## Trainer

Deterministic spots (`poker_alpha/analysis/trainer.SPOT_LIBRARY`) whose
coverage labels are verified against real engine output by tests. Grading:
accepted mixes → frequency language ("Common in solver mix — 64%"),
rollout decisions → estimated decision regret, uncertain solver states →
"PokerAlpha uncertain — review only" (no penalty). Session stats stay in
the browser session.
