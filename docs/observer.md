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

**Not validated.** No real screenshots have been collected, so no real
accuracy exists. `tests/fixtures/pokernow/` holds the structure (`raw/`,
`annotations/`, optional `calibration.json`) and the annotation format
(`pokeralpha.screenshot_annotation/v1`, see its README). Once screenshots and
annotations are added:

```bash
python experiments/observer_validation.py --fixture-dir tests/fixtures/pokernow
```

reports hero-card, board-card, stack (exact + MAE), bet (exact + MAE), pot
(exact + MAE), dealer, seat-occupancy and full-state accuracy and writes
`results/validation/observer_fixture_validation.json`. With an empty
directory it reports that nothing was measured.

## Adding real fixtures

1. Save screenshots of the real client (four-colour deck on) under
   `tests/fixtures/observer/` with a JSON ground truth per image (same layout
   as `tests/fixtures/table.json`).
2. Adjust a copy of `tests/fixtures/table_calibration.json` until
   `PokerNowStyleAdapter.extract_regions` crops the right areas.
3. Build `TemplateOCR.from_samples` from cropped glyphs, or install
   Tesseract.
4. Run `observer.evaluation.evaluate` on the fixtures and record the result
   — only those numbers say anything about the real client.
