"""Phase 52: explicit capture geometry (points vs pixels, Retina, offsets)."""

import pytest

from poker_alpha.observer.geometry import CaptureGeometry, suggested_crop

MAIN = {"left": 0, "top": 0, "width": 1512, "height": 982}          # MBP 14" in points
LEFT = {"left": -1920, "top": 0, "width": 1920, "height": 1080}     # external, left


def test_retina_2x():
    g = CaptureGeometry(MAIN, (100, 50, 640, 400), (1280, 800))
    assert g.global_box == (100, 50, 640, 400)
    assert g.pixels_per_point == (2.0, 2.0) and "Retina" in g.scale_label
    assert g.image_to_screen(0, 0) == (100, 50)
    assert g.image_to_screen(1280, 800) == (740, 450)
    assert g.screen_to_image(740, 450) == (1280, 800)


def test_non_retina_whole_monitor():
    g = CaptureGeometry(LEFT, None, (1920, 1080))
    assert g.global_box == (-1920, 0, 1920, 1080) and g.pixels_per_point == (1.0, 1.0)
    assert g.image_to_screen(10, 10) == (-1910, 10)


def test_fractional_scaling_and_round_trip():
    g = CaptureGeometry(LEFT, (200, 100, 800, 600), (1200, 900))
    assert g.pixels_per_point == (1.5, 1.5)
    x, y = g.image_to_screen(333, 222)
    assert g.screen_to_image(x, y) == pytest.approx((333, 222))
    assert g.global_box[0] == -1720                     # monitor-relative -> global


def test_rect_semantics_clipping_and_errors():
    g = CaptureGeometry(MAIN, (1400, 900, 400, 400), (224, 164))   # clipped to the monitor
    assert g.global_box == (1400, 900, 112, 82) and g.pixels_per_point == (2.0, 2.0)
    with pytest.raises(ValueError, match="outside the monitor"):
        CaptureGeometry(MAIN, (1600, 10, 100, 100), (10, 10)).global_box


def test_non_uniform_scale_is_flagged():
    g = CaptureGeometry(MAIN, (0, 0, 600, 400), (1200, 600))
    assert "non-uniform" in g.scale_label


def test_suggested_crop_maps_pixels_back_to_points():
    g = CaptureGeometry(MAIN, None, (3024, 1964))                   # full Retina monitor
    rect = suggested_crop(g, table_box=(1000, 600, 2200, 1200),
                          regions_box=(980, 580, 2220, 1400), margin=0.0)
    assert rect == (490, 290, 620, 410)                            # points, monitor-relative
