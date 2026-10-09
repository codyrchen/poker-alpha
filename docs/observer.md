# Screen observer (`poker_alpha.observer`, optional)

> Exact PokerNow visual accuracy is not validated without representative
> screenshots. Every accuracy number below is measured on **synthetic**
> frames produced by `observer/synthetic.py`.

The observer turns screenshots into an `ObservedTableState`. It only reads
pixels: there is no clicking, mouse or keyboard control, browser automation
or action submission anywhere in PokerAlpha. Use real-time observation only
in private, play-money or test games where assistance is permitted; where
it is not, process saved screenshots after the session.

Install with `pip install -e ".[vision]"` (Pillow, mss); Tesseract OCR is a
further optional backend (`[ocr]` + the `tesseract` binary).

## Pipeline

```
ScreenSource ─► locate_table ─► regions ─► OCR / card recognizer ─► FrameObservation
   (mss, files)   (calibrated bbox                                   (value, confidence,
                   or felt colour)                                     region, time)
                                     ─► FieldTracker smoothing ─► StateTracker rules
                                     ─► TrackedTableState ─► ObservedTableState ─► decision engine
```

* **Sources** — `MSSScreenSource` (live, read-only), `ImageFileSource`,
  `ImageSequenceSource`.
* **Calibration** — `TableCalibration` (`pokeralpha.calibration/v1` JSON):
  regions in *table-normalized* coordinates (resolution independent), felt,
  text, highlight, button, card-back and per-suit colours. The table box is
  fixed or detected from the felt colour each frame.
  `pokernow.default_layout(num_seats, hero_seat)` is a generic oval layout,
  not measured from PokerNow.
* **OCR** — `TemplateOCR`: connected-component glyph segmentation, splitting
  of touching glyphs, rejection of edge-touching and bar-shaped intruders,
  shape + geometry matching, number-grammar fix for `.` vs `,`. Its
  confidence is the weakest glyph's match score — a score, not a calibrated
  probability. `TemplateOCR.from_samples` learns glyphs from real crops.
  `TesseractOCR` raises `OCRUnavailable` when unavailable; OCR is never faked.
* **Cards** — `TemplateCardRecognizer`: card-face presence, rank via
  `TemplateOCR`, suit from ink colour. Requires a **four-colour deck**; with
  two-colour decks suits are ambiguous (shape templates are future work).
* **Smoothing** — `FieldTracker`: readings below a confidence floor are
  ignored; a change needs high confidence or N agreeing frames; hero/board
  cards always need N agreeing frames; fields can be pinned manually.
* **Fusion** — `StateTracker`: board cards cannot change or disappear
  mid-hand and must appear in order; stacks only grow with a pot award or a
  new hand (otherwise held, flagged, and accepted only after persistent
  agreement); the dealer only moves between hands; new hands are detected
  from a confirmed button move with an empty board. `pause`, `correct`,
  `release`, `resume` keep OCR errors recoverable. `infer_events` names
  bets, stack changes, folds, board cards, bets swept into the pot and new
  hands.

## Synthetic validation

`python experiments/observer_validation.py --frames 30` →
`results/data/observer_synthetic_validation.csv`:

| seats | size | noise σ | cards | numeric | state (1 frame) | state (3 fused frames) | false events/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1280×800 | 0 | 100% | 100% | 100% | 100% | 0 |
| 3 | 1280×800 | 0 | 100% | 100% | 100% | 90% | 0 |
| 6 | 1280×800 | 0 | 100% | 100% | 100% | 70% | 0 |
| 6 | 1280×800 | 8 | 100% | 97.4% | 73% | 70% | 0.11 |
| 6 | 1920×1200 | 0 | 100% | 100% | 100% | 100% | 0 |
| 9 | 1280×800 | 0 | 100% | 98.4% | 73% | 80% | 0 |
| 9 | 1600×1000 | 4 | 100% | 98.4% | 73% | 90% | 0 |
| 9 | 960×600 | 0 | 100% | 91.5% | 23% | 0% | 0 |

Reading the table honestly: fused accuracy can be *below* single-frame
accuracy because correct readings whose match score falls under the
confidence floor are deliberately never auto-accepted (they need a manual
correction); small text (9 seats at 960×600) breaks the template OCR; and
heavy noise produced one confident misread and false events. Wrong
readings usually carry much lower confidence than right ones, which is what
the tracker relies on — usually, not always.

## Real PokerNow validation status

**One real heads-up frame, used for alignment and tuning; not validated.**
`tests/fixtures/pokernow/raw/hu_preflop_0001.png` is a real PokerNow
heads-up table (preflop, blinds 0.25 / 0.50), cropped to the table, with the
chat preview and both player names painted over; its ground truth is in
`annotations/hu_preflop_0001.json` and the layout in `calibration.json`.
The harness scores it fully correct (`results/validation/
observer_fixture_validation.json`), but the same frame was used to place
the regions and to choose the PokerNow OCR / card-recognizer settings, so
this is a regression check, **not** an accuracy measurement. The note in the
JSON says so. Real accuracy needs more annotated frames, especially ones
not used for tuning (flop / turn / river boards, other ranks and suits,
all-in, folded, larger stacks).

```bash
python experiments/observer_validation.py --fixture-dir tests/fixtures/pokernow
```

reports hero-card, board-card, stack (exact + MAE), bet (exact + MAE), pot
(exact + MAE), dealer, seat-occupancy and full-state accuracy and writes
`results/validation/observer_fixture_validation.json`. With an empty
directory it reports that nothing was measured.

## PokerNow heads-up preset

`pokernow_hu_layout(hero_side="right")` (UI: *Layout -> PokerNow
Heads-Up*) replaces the generic oval for real PokerNow heads-up tables.
The generic `default_layout(2, 0)` puts seat 1 at the top of the oval; in
PokerNow heads-up **both players sit along the bottom edge**, so the generic
layout reads both seats as empty and misses the hero cards, stacks and
dealer button.

| | |
| --- | --- |
| seats | seat 0 = hero (bottom-right by default), seat 1 = opponent (bottom-left); `hero_side="left"` mirrors |
| per player | hole cards (hero faces / opponent backs) on the left of the name + stack plate, the street bet as a "+1.00" pill under the stack, dealer button above the card / plate junction, the plate turns pale yellow for the player to act |
| table bounds | `table_detector="green_oval"`: ellipse fitted (by moments) to the largest felt-hue blob, on a frame subsampled to ~640 px. It follows window size and browser zoom and ignores the felt's shading, the logo watermark, the pot pill and badges touching the felt edge |
| pot | PokerNow's centre number leaves out bets still in front of players (`pot_includes_bets=False`); the tracker adds visible bets to get the total pot the state model expects |
| board | 5 slots in the felt centre, **not yet verified** (the reference frame is preflop) |
| recognizers | `client="pokernow"`: amounts via `pokernow_ocr()` (DejaVu Sans Bold templates, 0.5 relative threshold, histogram-mode background so bet pills inside plates work, separators / "+" not capping confidence); cards via `PokerNowCardRecognizer` |

`PokerNowCardRecognizer` handles PokerNow's two-colour deck and dimmed
cards: the card face is found relative to its own brightness (the hero's
cards in the reference frame are dimmed to grey ~100), rank = upper-left ink
matched against DejaVu Serif Bold glyphs rotated +-12 degrees (hole cards
are fanned), suit = red / black by colour, then spade vs club or heart vs
diamond by shape. Only J of spades and 7 of hearts have been seen on a real
frame; the other 11 ranks and 2 suits are untested on real PokerNow images.

On the reference frame all fields match the ground truth at 0.8x-3x scale
and on the uncropped desktop screenshot, at 45-270 ms per frame. Confidence
values are match scores: hero cards ~0.4-0.55, amounts ~0.4-0.6, so the
fused critical confidence is ~0.40 and the live mode's default threshold of
0.5 blocks decisions on this frame. Lowering the threshold is a judgement
call: it accepts match scores that are not calibrated probabilities.

## Live screen mode (Streamlit)

`MSSScreenSource -> PokerNowStyleAdapter -> StateTracker -> ObservedTableState
-> DecisionReport`, implemented in `poker_alpha/observer/live.py` (no
Streamlit dependency, unit-tested with a mocked screen) and
`poker_alpha/ui/live_panel.py`.

Read-only: frames are captured and analysed on this machine, never
uploaded, and written to disk only when **Save current frame** is pressed
(default folder `~/pokeralpha_captures`, PNG plus a small JSON sidecar).
Nothing clicks, types, controls a browser or submits actions. Use live
mode only in private, play-money or test games where real-time assistance
is permitted; otherwise save frames / screenshots and analyse after the
session.

```bash
pip install -e ".[vision,ui]"
streamlit run poker_alpha/ui/app.py
```

Sidebar -> Input -> **Live screen**.

**1. Capture.** Pick the monitor (mss index; 1 = main display). Set the
capture rectangle `left / top / width / height` relative to that monitor in
screen points (macOS screenshot coordinates: press Cmd+Shift+4 and read the
numbers next to the crosshair at the table's top-left and bottom-right
corners). Width or height 0 captures the whole monitor. On Retina displays
the captured frame is 2x the point size; the raw-frame caption shows the
pixel size.

**2. Calibrate** (open "Calibration / debug workflow"; untick *Compute
decisions* while doing this):
1. Press **Capture one frame**. The left image is the raw frame, the right
   one the overlay: yellow = table bounds, magenta = hero cards, cyan =
   board, orange = pot, green = stacks, red = bets, white = dealer-button
   spots, blue = seat name / card-back / highlight boxes; labels show the
   last raw reading.
2. Pick the layout: **PokerNow Heads-Up** (choose which side the hero's
   plate is on, then **Use PokerNow Heads-Up layout**) or **Generic
   layout** (seats, hero seat, **Use default layout**).
3. Table bounds: *PokerNow felt (hue)* (set by the PokerNow preset; follows
   the felt automatically), *Fixed box* (x0, y0, x1, y1 in pixels of the
   captured frame) or *Detect felt colour* (pick the felt colour or "Sample
   felt colour at table centre", adjust tolerance).
4. Coarse alignment: change dx / dy / scale x / scale y and press **Apply**
   until the boxes sit on the table elements.
5. Fine-tune single regions (x, y, w, h in table-normalized units) and press
   **Update**. Compare each box with the *Raw recognition* table.
6. **Save calibration** to a JSON path; **Load calibration** restores it.
   Changing the calibration resets the tracker.

**3. Run.** Set the blinds and frames per second (0.5-3, default 1), tick
*Compute decisions*, press **Start Live Observer** (Stop to end). The same
`StateTracker` persists across frames in `st.session_state`; the loop is a
Streamlit fragment rerun on a timer (`run_every`), never a blocking loop.
The page shows the raw per-field readings and confidences of the last frame
separately from the fused tracker state (hero cards, board, pot, dealer,
actor, per-seat occupied / folded / all-in, stacks, bets, confidences) and
the critical confidence. A decision is computed only when the critical
check passes: hero cards confirmed, pot and hero stack read, dealer found,
critical confidence >= the threshold (default 0.5) and no rules-level
validation error. Otherwise the reasons are shown and no decision is made.
The report is recomputed only when the observed state changes.

Capture failures (missing permission, display changes, mss missing) are
shown in the page and the observer keeps running; an all-uniform frame
triggers a Screen Recording permission hint.

## Annotation validation and metrics (Phase 38)

`observer.annotations.validate_annotation` rejects inconsistent ground
truth before it is used (card syntax, duplicate cards, board length vs
`street`, seat ranges, hero seat occupied, non-negative stacks/bets/pot,
`all_in` with a non-zero stack). Annotations may carry `street`, per-seat
`all_in` and `name`. Scoring reports hero-card and board-card accuracy per
card and as exact pair/board, street, stacks / bets / pot (exact + MAE),
dealer, seat occupancy, full state, and accuracy by reported confidence
(calibration). A harness self-test runs it end to end on the synthetic image
(`tests/test_annotation_validation.py`) — that proves the plumbing, not real
accuracy. **Real screen validation: BLOCKED** (0 real fixtures, Phase 38).

## Adding real fixtures (unblocks Phase 30)

1. Capture screenshots of the real client only where you are allowed to
   (your own private or play-money games; four-colour deck on) into
   `tests/fixtures/pokernow/raw/`, and write one
   `pokeralpha.screenshot_annotation/v1` JSON per image into
   `tests/fixtures/pokernow/annotations/` (format in that directory's
   README).
2. Tune a copy of `tests/fixtures/table_calibration.json` as
   `tests/fixtures/pokernow/calibration.json` until
   `PokerNowStyleAdapter.extract_regions` crops the right areas.
3. Build `TemplateOCR.from_samples` from cropped glyphs, or install
   Tesseract.
4. Run `python experiments/observer_validation.py --fixture-dir
   tests/fixtures/pokernow` and record the result — only those numbers say
   anything about the real client. A few dozen screenshots across seat
   counts, streets, all-in and folded states are the minimum for a
   meaningful per-field accuracy.

## Region debugger and debug reports (Phase 47)

Live screen -> **Region debugger** (expander under the readings). Tabs
Hero / Board / Seats / Pot/Bets / Dealer/Actor / Tracker. For every
recognition region: the crop (2x), pixel box and normalized box, and for
each field read from it the raw value, confidence, fused value, the
tracker's decision for this frame (`accepted` with high confidence or after
N agreeing frames, `pending` k/N, `agrees`, `rejected` below the minimum
confidence or by a tracker rule such as a board-card change, `held` stack
increase without a pot award, `pinned`), the previously accepted value and
the agreeing-frame count. The Tracker tab lists every field tracker and
the recent tracker flags.

The explanation is derived by comparing each field tracker's state before
and after the update (captured by `LiveObserverSession.step`); it does not
change recognition or tracking. **Export current debug report** writes
`<save folder>/debug/debug-<time>-f<frame>.json` and a self-contained
`.html` with the crops embedded (local, never overwritten, folder carries a
`*` .gitignore). Code: `poker_alpha/observer/debug.py`.

## Calibration files (format v2, Phase 53)

`TableCalibration.save` writes `pokeralpha.calibration/v2`: schema version,
name, seats, hero seat, table detector + settings (felt colour /
tolerance / fixed box), recognizer client, pot convention, all colours,
the normalized regions, a `region_semantics` note, provenance (`source`,
`created`, `updated`, `migrated_from`) and `checksum` = SHA-256 of the
geometry payload (everything that changes what the observer reads;
provenance and timestamps excluded, so re-saving does not change it).

Loading: v2 is verified against its checksum (a hand-edited or damaged
geometry is refused: re-save from the UI or `load(path, verify=False)`
deliberately); unknown fields and newer formats are refused; v1 files are
migrated with the documented v1 defaults (felt-colour detector, generic
client, pot including bets — exactly what v1 meant) and marked
`migrated_from: v1`. Geometry is never silently reinterpreted. Sessions
identify calibrations by the geometry checksum. Tests:
`tests/test_calibration_versioning.py`.
