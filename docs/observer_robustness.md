# Observer robustness matrix (Phase 49)

**SYNTHETIC / DERIVED ROBUSTNESS ONLY — this is not real PokerNow accuracy.**

`python experiments/observer_robustness.py` (seed 0, 8 synthetic
tables of 2 / 6 / 9 seats, 112 s) ->
`results/validation/observer_robustness_v1.json`. Each perturbation is
applied to the synthetic tables (exact ground truth) and to the single real
PokerNow frame. The real frame is the TUNING frame, so its columns are
labelled **DERIVED FROM TUNING FRAME — NOT INDEPENDENT VALIDATION**: they show
how the PokerNow recognizers degrade, not how accurate they are.

Columns: table found (fraction), alignment = IoU of the located table box
with the expected one, cards / amounts = fraction of card / amount fields
equal to the truth, recovery = clean frames needed after 3 perturbed ones
until `live_check` passes (2-seat synthetic tables), confirmed-state change
= did any confirmed critical value (hero cards, pot, stacks) change during
the 3-frame burst.

| perturbation | level | synth found | synth IoU | synth cards | synth amounts | recovery | confirmed state changed | DERIVED found | DERIVED cards | DERIVED amounts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scale | 0.5 | 1.00 | 1.00 | 1.00 | 0.72 | 1 | no | 1.00 | 1.00 | 0.20 |
| scale | 0.67 | 1.00 | 1.00 | 1.00 | 0.92 | 1 | no | 1.00 | 1.00 | 0.40 |
| scale | 0.8 | 1.00 | 1.00 | 1.00 | 0.76 | 1 | no | 1.00 | 1.00 | 1.00 |
| scale | 1.25 | 1.00 | 1.00 | 1.00 | 0.89 | 1 | no | 1.00 | 1.00 | 1.00 |
| scale | 1.5 | 1.00 | 1.00 | 1.00 | 0.92 | 1 | no | 1.00 | 1.00 | 1.00 |
| scale | 2.0 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 1.00 |
| brightness | 1.25 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 0.80 |
| brightness | 1.5 | 0.00 | — | — | — | 1 | no | 1.00 | 1.00 | 0.80 |
| brightness | 0.7 | 1.00 | 1.00 | 0.34 | 0.99 | 2 | no | 1.00 | 0.71 | 0.80 |
| brightness | 0.5 | 0.00 | — | — | — | 1 | no | 1.00 | 0.71 | 0.20 |
| contrast | 0.7 | 1.00 | 1.00 | 0.34 | 0.99 | 2 | no | 1.00 | 1.00 | 0.80 |
| contrast | 0.5 | 1.00 | 1.00 | 0.34 | 0.99 | 2 | no | 1.00 | 0.71 | 0.40 |
| contrast | 1.3 | 1.00 | 1.00 | 1.00 | 0.80 | 1 | no | 1.00 | 1.00 | 0.80 |
| blur | 0.5 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 1.00 |
| blur | 1.0 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 0.20 |
| blur | 1.5 | 1.00 | 1.00 | 0.96 | 0.62 | 1 | no | 1.00 | 1.00 | 0.20 |
| blur | 2.0 | 1.00 | 1.00 | 0.91 | 0.42 | 1 | no | 1.00 | 1.00 | 0.20 |
| gaussian_noise | 4 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 1.00 |
| gaussian_noise | 8 | 1.00 | 1.00 | 1.00 | 0.97 | 1 | no | 1.00 | 1.00 | 0.80 |
| gaussian_noise | 16 | 1.00 | 1.00 | 0.91 | 0.94 | 1 | no | 1.00 | 0.86 | 0.40 |
| jpeg_quality | 80 | 1.00 | 1.00 | 1.00 | 0.98 | 1 | no | 1.00 | 1.00 | 0.80 |
| jpeg_quality | 50 | 1.00 | 1.00 | 1.00 | 0.98 | 1 | no | 1.00 | 1.00 | 1.00 |
| jpeg_quality | 30 | 1.00 | 1.00 | 1.00 | 0.95 | 1 | no | 1.00 | 1.00 | 0.80 |
| jpeg_quality | 15 | 1.00 | 1.00 | 1.00 | 0.92 | 1 | no | 1.00 | 1.00 | 0.60 |
| crop_offset | 5 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 0.00 | — | — |
| crop_offset | 25 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 0.00 | — | — |
| crop_offset | 80 | 0.00 | — | — | — | 1 | no | 0.00 | — | — |
| rotation_deg | 0.5 | 1.00 | 0.98 | 0.96 | 0.96 | 1 | no | 1.00 | 1.00 | 0.80 |
| rotation_deg | 1.0 | 1.00 | 0.97 | 0.95 | 0.69 | 1 | no | 1.00 | 1.00 | 1.00 |
| rotation_deg | 2.0 | 1.00 | 0.94 | 0.75 | 0.51 | 1 | no | 1.00 | 1.00 | 0.40 |
| translation | 40 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 1.00 |
| translation | 150 | 1.00 | 1.00 | 1.00 | 0.99 | 1 | no | 1.00 | 1.00 | 1.00 |
| browser_zoom | 0.9 | 1.00 | 1.00 | 1.00 | 0.90 | 1 | no | 1.00 | 1.00 | 1.00 |
| browser_zoom | 0.8 | 1.00 | 1.00 | 1.00 | 0.76 | 1 | no | 1.00 | 1.00 | 1.00 |
| browser_zoom | 1.1 | 1.00 | 1.00 | 0.96 | 0.97 | 1 | no | 0.00 | — | — |

## Reading it

* **No confirmed state changes under any perturbation** (column 8): degraded
  frames produce misreads, but the tracker holds them and `live_check`
  refuses decisions until a frame agrees again; recovery takes 1-2 clean
  frames.
* The first run of this matrix found two more fail-unsafe paths, fixed in
  Phase 49 (`docs/observer_failure_modes.md`): pot misreads (479.5 -> 479.6,
  479.5 -> 4795) were confirmed after 2 frames, and a table cut off by the
  capture edge mis-aligned every region (pot read 0, a seat "folded and
  left"). Now: an unexplained pot change is held (accepted after 4 agreeing
  frames, flagged) and a change breaking chip conservation by > 25% (raw
  readings of the same frame) is never auto-accepted; a felt touching the
  capture edge is reported as "table cut off — widen the capture".
* Table detection is the most fragile step for the generic colour-tolerance
  detector (brightness x1.5 / x0.5 -> not found -> stale, no decision).
  The PokerNow hue detector found the table in every derived case.
* Amount OCR is the weakest reader: synthetic amounts drop to 0.42-0.76 at
  blur 2 px, rotation 2 degrees, 0.5x / 0.8x scale; card reading holds up
  better. Misreads lower confidence on average (`conf_delta` in the JSON) but
  confidence alone does not separate them — the tracker's plausibility rules
  do.
* Derived (tuning-frame) amounts are low at 0.5x / 0.67x scale and blur >= 1:
  the real frame is a downscaled screenshot (digits ~9 px tall) — a reason
  to capture at native Retina resolution, not evidence about real accuracy.
