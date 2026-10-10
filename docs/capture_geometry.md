# Capture geometry: points, pixels, table box, regions (Phase 52)

Code: `poker_alpha/observer/geometry.py` (`CaptureGeometry`), tests:
`tests/test_capture_geometry.py`.

| system | meaning | unit |
| --- | --- | --- |
| monitor | one display as mss reports it: `left, top, width, height`; `left/top` are global (a display left of the main one has a negative `left`) | screen points |
| capture rectangle | `(left, top, width, height)` **relative to the selected monitor's top-left**; `None` or width/height 0 = the whole monitor; a rectangle extending past the monitor is clipped; one *starting* outside it is an error | screen points |
| global capture box | monitor origin + capture rectangle = what mss grabs (`live.absolute_box`) | screen points |
| captured image | the frame mss returns | pixels |
| table box | `(x0, y0, x1, y1)` of the located table in the captured image, or a fixed `table_bbox` calibration | pixels |
| normalized region | `Region(x, y, w, h)` as fractions of the table box | 0..1 of the table box |

## Retina and display scaling

macOS screen coordinates — including the numbers Cmd+Shift+4 shows next to
the crosshair — are **points**. A Retina display returns **2 pixels per
point**: a 640x400-point capture rectangle gives a 1280x800-pixel image.
Windows / Linux display scaling can give 1.25 or 1.5. `CaptureGeometry`
computes pixels-per-point from the captured image size and the global box,
and the live page shows it next to the raw frame ("2.00 px/pt (Retina)"). A
non-uniform ratio is flagged as unexpected.

Consequences:

* enter the capture rectangle in **points** (Cmd+Shift+4 numbers), never in
  the pixel sizes shown under the captured image;
* normalized regions are resolution-independent, so the same calibration
  works at 1x, 1.5x and 2x and at any browser zoom (the table is located
  every frame);
* a **fixed** `table_bbox` is in captured-image pixels and is only valid for
  the capture rectangle and scale it was set with — prefer felt / hue
  detection;
* the recorder stores `geometry` (monitor, rectangle, global box, image size,
  pixels per point) with every kept sample, so a session can be re-mapped
  later.

When the located table covers well under a third of the capture, the live
page suggests a tighter capture rectangle (monitor-relative points around
the table and all recognition regions plus 4% margin). It is a hint only;
nothing is changed automatically.
