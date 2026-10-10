# PokerAlpha product UI v0.1 — Play Mode specification

The app now has two top-level modes (sidebar → **Mode**):

* **Play** — the user-facing decision view (this spec).
* **Developer** — the original engineering interface, unchanged: manual
  JSON, screenshot observer, full live panel (calibration editor, overlay,
  OCR/fused tables, region debugger, test sessions), annotation tools, and
  the complete diagnostic report.

Play Mode's defining UX feature mirrors the research: **PokerAlpha knows
when not to trust itself**, and the interface treats abstention as a
first-class state, not an error.

## Information hierarchy

| tier | content | where |
|---|---|---|
| primary | hero cards + board, pot / to-call / effective stack / SPR, the action list with frequencies, the recommended action, solver confidence | always visible, top of page |
| secondary | method (solver / rollout / heuristic), EV edge, equity ± SE, pot odds | status strip under the actions |
| advanced | gate signals (visits, seed disagreement, movement, collision), source cascade, uncertainty map, opponent ranges, OCR internals | "Why this recommendation" → "Technical details" expander, or Developer Mode |

## States

**Normal recommendation** — card panel; action rows (frequency-descending,
stable ties) with a frequency bar, percentage and EV ± SE where measured;
the recommended row carries an accent border and a RECOMMENDED tag; status
strip with confidence badge (green only at HIGH), method, EV edge, equity,
pot odds; a "Why this recommendation" expander explaining source, gate
status and EV comparison in plain language, with the full tables one level
deeper.

**Abstention (solver withheld)** — when the confidence gate rejects the
lookup: an amber-edged panel titled "Solver recommendation withheld" with
the *actual* gate reasons translated to plain language (e.g.
"independently trained runs disagree here") and the fallback that produced
any advice shown below it. Raw reason codes stay available in the
technical expander. Low-confidence (downgraded, not withheld) lookups get
the same panel in its non-withheld variant above the solver's frequencies.

**Refusal / waiting** — when the state itself cannot be trusted
(`refusal_codes`, failed live validation): a neutral panel with the
smallest useful human sentence ("Waiting for a reliable pot read", "It is
not the hero's turn"); no stale recommendation is ever shown. Technical
problems expand below.

## Live state confidence

Live view separates **Recognition** (critical-state confidence from the
observer) from **Solver trust** (the gate's verdict) as two badges. If
recognition fails validation, the recommendation area shows the waiting
state — never a stale report. The Play live view shows only: capture
setup (collapsed once running), the two badges, the current hand, and the
recommendation; raw frames, overlays, OCR tables and the region debugger
remain Developer-Mode-only.

## Demo hands (zero setup)

Play → Demo offers five named spots, each executed through the real
pipeline (release strategy + gate + rollouts; nothing pre-recorded), with
labels that state what actually happens:

1. River, checked to hero — solver accepted
2. River, facing a bet — solver at low confidence (off-tree size)
3. Preflop open decision — gate rejects (seed disagreement)
4. Preflop vs 3-bet — rollout fallback
5. 6-max flop c-bet — multiway (rollout only)

60-second path: `./scripts/run_live_observer.sh` (or
`streamlit run poker_alpha/ui/app.py`) → the app opens in Play → Demo with
the first spot already analysed → switch spots from the dropdown.

## Future compact overlay

`poker_alpha.ui.viewmodel.compact_summary(report)` produces the minimal
model a later always-on-top overlay needs — up to three action lines with
frequencies plus one confidence line, or the withheld state:

```
Bet 6.2 BB       64%
Check            18%
Bet 9.4 BB       18%

HIGH CONFIDENCE
```

No desktop shell is built in this phase.

## Code layout

| file | role |
|---|---|
| `poker_alpha/ui/viewmodel.py` | pure view models (no Streamlit): actions, abstention, refusal, hand summary, live status, why-panel, compact overlay summary, reason-code → human text |
| `poker_alpha/ui/components.py` | Streamlit renderers for those models |
| `poker_alpha/ui/styles.py` | the single CSS block (dark-neutral panels, one accent `#2a78d6`, amber caution, green only for HIGH) |
| `poker_alpha/ui/play.py` | Play page: Demo / Manual / Hand history / Live inputs |
| `poker_alpha/ui/app.py` | mode switch; Developer Mode unchanged |
| `poker_alpha/ui/view.py` | original table helpers (still used by Developer Mode and the technical expander) |

Tests: `tests/test_play_ui.py` (view-model transformations + headless
AppTest for the demo, abstention and Developer-mode preservation).

## Visual snapshot

`docs/ui_preview.html` is a static capture of the **real markup** the app
emitted for the three key states (normal / abstention / low-confidence),
generated headlessly through the actual pipeline — open it in any browser.
Regenerate by re-running the snippet in the git history of this commit or
simply by launching the app. Native OS screenshots were not captured in
this environment (screen-recording permission is not granted to the build
shell); the preview file and the headless DOM assertions are the review
artifacts.

## Style rules (kept deliberately boring)

One accent color; status colors only with semantic meaning; no gradients,
glow, emoji walls or casino imagery; cards rendered as typographic chips
(no image assets); flex layouts that wrap at narrow widths — the
recommendation column reads fine at ~700 px.
