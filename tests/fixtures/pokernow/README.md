# Real PokerNow screenshot fixtures (currently EMPTY)

This directory is the place for **real** screenshots and human annotations.
It intentionally contains no images: none have been collected, so **no real
PokerNow accuracy has been measured**. Do not put synthetic images here.

Use only screenshots you are allowed to capture (e.g. your own private or
play-money games). The observer is read-only; nothing here interacts with
PokerNow.

## Layout

```
tests/fixtures/pokernow/
    README.md
    calibration.json        # optional; a TableCalibration tuned to the client
    raw/                    # screenshots: <name>.png / .jpg
    annotations/            # one <name>.json per screenshot
```

## Annotation format (`pokeralpha.screenshot_annotation/v1`)

```json
{
  "format": "pokeralpha.screenshot_annotation/v1",
  "num_seats": 6,
  "hero_seat": 3,
  "dealer_seat": 3,
  "hero_cards": ["As", "Kd"],
  "board": ["Qs", "Jh", "4c"],
  "pot": 13.5,
  "seats": [
    {"seat": 0, "stack": 97.5, "bet": 2.5, "active": true},
    {"seat": 3, "stack": 100.0, "bet": 0, "active": true}
  ]
}
```

* `seats` lists occupied seats only; missing seats are empty.
* `active` = still in the hand (not folded).
* Amounts are as displayed by the client. `pot` is the displayed pot.
* Cards use rank + suit letters (`T` for ten).
* `"image": "raw/other_name.png"` overrides the default `raw/<name>.<ext>`.

## Running

```bash
python experiments/observer_validation.py --fixture-dir tests/fixtures/pokernow
```

Reports hero-card, board-card, stack (exact + MAE), bet (exact + MAE), pot
(exact + MAE), dealer, seat-occupancy and full-state accuracy, and writes
`results/validation/observer_fixture_validation.json`. With no annotated
screenshots it says so and reports nothing as measured.

Without `calibration.json` the generic `default_layout(num_seats, hero_seat)`
is used, which was **not** derived from PokerNow and will likely need
recalibration (see `docs/observer.md`).
