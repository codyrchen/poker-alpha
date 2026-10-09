# Real PokerNow screenshot fixtures

This directory holds **real** PokerNow screenshots and human annotations;
never put synthetic images here.

| frame | content | note |
| --- | --- | --- |
| `hu_preflop_0001` | heads-up NLH 0.25 / 0.50, preflop, hero bottom-right (Js 7h, 49.25), opponent bottom-left (49.75, bet 1.00, dealer), pot display 0.00 | cropped to the table; chat preview and player names painted over. Also the frame the heads-up layout (`calibration.json`) was aligned on and the PokerNow recognizers were tuned on |

Because of that, scoring this directory is a regression check, **not** an
accuracy measurement (`NOTE.txt` puts that caveat into the harness
output). Add frames that were not used for tuning to measure accuracy.

Use only screenshots you are allowed to capture (e.g. your own private or
play-money games). Crop them to the table and paint over anything personal
(names, chat, other windows) before committing. The observer is read-only;
nothing here interacts with PokerNow.

## Layout

```
tests/fixtures/pokernow/
    README.md
    calibration.json        # TableCalibration (the PokerNow heads-up preset)
    NOTE.txt                # caveat copied into the harness output
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

`calibration.json` is `pokernow_hu_layout()` (2 seats). Without it the
generic `default_layout(num_seats, hero_seat)` is used, which was **not**
derived from PokerNow (see `docs/observer.md`).
