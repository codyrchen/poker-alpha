# Observer failure modes and fail-safe behaviour (Phase 48)

Tests: `tests/test_observer_failures.py` (synthetic frames, fake screen; no
real PokerNow data). Required behaviour for every bad frame:

* never raise; report an error / warning where one applies;
* **never replace a confirmed value with a misread one** (no hallucinated
  table state) — the last valid state is preserved;
* refuse a decision while the latest frame is unusable (`stale`) or
  disagrees with the confirmed critical state (`state not settled`).

## Decision gate (`live_check`)

`critical_check` (hero cards confirmed, pot, hero stack, dealer, critical
confidence, rules-level validation) **plus**:

| condition | problem reported |
| --- | --- |
| the latest frame could not be captured or the table not found | `stale: the latest frame was not recognized (...); N bad frame(s) in a row` |
| the latest frame's raw reading of a critical field (hero cards, board, pot, hero stack, bets of seats in the hand) differs from the confirmed value | `state not settled: the latest frame disagrees with the confirmed state on ...` |

The live UI uses `live_check`; a decision reappears as soon as a frame is
read that agrees with the confirmed state.

## Tracker plausibility rules (added in Phase 48)

Found by the failure injections below; each blocks a poker-impossible
transition caused by a bad frame, holds it (flagged) and accepts it only if
it persists:

| rule | before the fix | now |
| --- | --- | --- |
| a cleared board is a new hand only if **new hero cards are visible** | an unreadable frame (every card "absent") started a phantom new hand and wiped the board and hero cards | new hands come from a dealer move, or a cleared board with new visible hero cards |
| a confirmed hero card cannot change or vanish within a hand | a dark box over the suit turned Kd into Ks after 3 frames; colour / brightness shifts erased the hero cards | held (`held hero_card_N change`), accepted only after 6 agreeing frames (flagged) |
| a bet shrinks only within 3 frames of a collection (pot rose / board grew) or at a new hand | a blurred "2" read as "0" (score 0.81) erased a bet | held, accepted after 4 frames (flagged) |
| a seat in the hand cannot vacate / lose its stack mid-hand | an occluded stack became "empty" | held, accepted after 4 frames (flagged) |
| the detected table must be table-shaped (aspect 1.0-3.5) | a dark-grey browser tab bar matched the felt colour; regions read "empty" and the pot dropped to 0 | `CalibrationError: felt-coloured region ... is not table-shaped` -> stale, no decision |

Board-card changes, dealer moves with cards out and unexplained stack
increases were already held / rejected. New hands still update cards
without extra delay (candidate counts survive the new-hand reset).

## Scenario results

| scenario | frame usable? | confirmed state changed? | decision |
| --- | --- | --- | --- |
| black / white frame, blank browser, tab switched away, table not found | no (table not found) | no | refused (stale) |
| PokerAlpha captures itself (thumbnail) | no (felt too small) — error suggests self-capture | no | refused |
| modal over table | no | no | refused |
| capture rectangle outside the monitor | error `starts outside the monitor` (was: silent 1-px strip) | no | refused |
| zero-width crop | = whole monitor (documented) | — | — |
| monitor disconnected / reconnected | capture error, source recreated next frame | no | refused until a frame is read |
| monitor dimensions change, browser moved / resized | yes | no | allowed |
| 50 / 80 / 125 / 150 % scale, Retina 2x | yes | no | allowed when the frame agrees |
| blur 1 px, JPEG q30, brightness +40 % | yes | no | allowed |
| blur 2 px, JPEG q10, brightness -40 %, contrast 50 %, warm / cool tint | misreads (cards "absent", digits wrong) | no | refused (not settled) |
| partial occlusion of hero cards / board | misreads | no | refused (not settled) |
| malformed / missing calibration | `CalibrationError` / `FileNotFoundError` / `ValueError` with a reason | — | — |
| OCR / Tesseract unavailable | `OCRUnavailable`; the default template OCR does not need Tesseract | — | — |
| malformed OCR output (`4?..,1`) | value unreadable, confidence 0 | no | refused |
| card crop blank, hero cards missing for 2 frames | held | no | — |
| duplicate cards (board card = hero card) | — | — | refused (state invalid) |
| impossible board change, stack increase mid-hand, dealer jump mid-hand | rejected / held with a flag | no | — |

The advisory warning "the table covers only x% of the capture ... is
PokerAlpha capturing its own window?" appears when a found table covers less
than 8% of the capture.

## Added in Phase 49 (found by the robustness matrix)

| rule | before | now |
| --- | --- | --- |
| a pot change needs evidence: bets moving, a new board card, a collection / bet change in the last 3 frames, or a drop to <= 2 BB (award / new hand) | a misread digit (479.5 -> 479.6) or lost decimal point (-> 4795) was confirmed after 2 frames | held, accepted after 4 agreeing frames (flagged) |
| a pot change breaking chip conservation (stacks + pot [+ bets], raw readings of the same frame) by > 25% | — | never auto-accepted (`rejected pot ...: breaks chip conservation`); correct it manually if real |
| the table felt must not touch the capture edge | a cut-off table mis-aligned every region (pot read 0, a seat "left") | `CalibrationError: table cut off ... widen the capture` -> stale, no decision |
