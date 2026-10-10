"""Phase 16: observer building blocks (sources, calibration, OCR, cards)."""

import json

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from poker_alpha.observer import CalibrationError, OCRUnavailable  # noqa: E402
from poker_alpha.observer.calibration import (TableCalibration,  # noqa: E402
                                              locate_table)
from poker_alpha.observer.cards import RANKS, TemplateCardRecognizer  # noqa: E402
from poker_alpha.observer.pokernow import default_layout  # noqa: E402
from poker_alpha.observer.regions import Region  # noqa: E402
from poker_alpha.observer.source import (ImageFileSource,  # noqa: E402
                                         ImageSequenceSource)
from poker_alpha.observer.synthetic import draw_card  # noqa: E402
from poker_alpha.observer.text import (TemplateOCR, TesseractOCR,  # noqa: E402
                                       fix_separators, parse_amount)

pytestmark = pytest.mark.vision


def text_image(txt, size=20, bg=(31, 94, 61), fg=(240, 240, 240)):
    img = Image.new("RGB", (max(60, size * len(txt)), size * 2), bg)
    ImageDraw.Draw(img).text((4, size // 3), txt, fill=fg,
                             font=ImageFont.load_default(size=size))
    return img


@pytest.mark.parametrize("size", [14, 18, 24, 36])
@pytest.mark.parametrize("txt", ["0.5", "13.5", "96.5", "100", "1,234.5",
                                 "7,890", "2.25", "1,000,000"])
def test_template_ocr_reads_amounts(size, txt):
    res = TemplateOCR().read_text(text_image(txt, size))
    assert fix_separators(res.text) == txt
    assert parse_amount(fix_separators(res.text)) == float(txt.replace(",", ""))
    assert 0.0 < res.confidence <= 1.0


def test_empty_region_and_unparseable_text():
    blank = Image.new("RGB", (80, 30), (31, 94, 61))
    assert TemplateOCR().read_text(blank).text == ""
    assert parse_amount("1,23,4") is None and parse_amount("") is None
    assert fix_separators("1.234.5") == "1,234.5"
    assert fix_separators("1.000") == "1,000"


def test_low_confidence_for_foreign_glyphs():
    # Letters outside the charset must not come back as confident digits.
    res = TemplateOCR().read_text(text_image("WXYZ", 24))
    assert res.confidence < 0.5


def test_tesseract_backend_never_fakes():
    try:
        TesseractOCR()
    except OCRUnavailable:
        pass  # expected on machines without tesseract
    else:  # pragma: no cover - only where tesseract exists
        pytest.skip("tesseract available; nothing to check")


@pytest.mark.parametrize("card", ["As", "Td", "7h", "2c", "Qs", "Kh", "9d", "Jc"])
@pytest.mark.parametrize("w,h", [(48, 80), (70, 120)])
def test_card_recognizer_reads_rendered_cards(card, w, h):
    cal = default_layout(6)
    img = Image.new("RGB", (w + 10, h + 10), cal.felt_color)
    draw_card(ImageDraw.Draw(img), (5, 5, 5 + w, 5 + h), card, cal)
    reading = TemplateCardRecognizer(cal.suit_colors).recognize(
        img.crop((5, 5, 5 + w, 5 + h)))
    assert reading.present and reading.card == card
    assert reading.confidence > 0.3


def test_card_slot_empty():
    cal = default_layout(6)
    empty = Image.new("RGB", (60, 100), cal.felt_color)
    r = TemplateCardRecognizer(cal.suit_colors).recognize(empty)
    assert not r.present and r.card is None and r.confidence > 0.9


def test_calibration_round_trip_and_validation(tmp_path):
    cal = default_layout(9, hero_seat=4)
    path = tmp_path / "cal.json"
    cal.save(path)
    again = TableCalibration.load(path)
    assert again.to_dict() == cal.to_dict()
    d = cal.to_dict()
    del d["regions"]["pot"]
    with pytest.raises(CalibrationError):
        TableCalibration.from_dict(d)
    with pytest.raises(CalibrationError):
        TableCalibration.from_dict({**cal.to_dict(), "format": "x"})
    with pytest.raises(ValueError):
        Region(0.1, 0.1, 0.0, 0.1)


def test_locate_table_by_felt_and_fixed_bbox():
    cal = default_layout(6)
    img = Image.new("RGB", (400, 300), (20, 20, 20))
    ImageDraw.Draw(img).rectangle((50, 40, 349, 259), fill=cal.felt_color)
    assert locate_table(img, cal) == (50, 40, 350, 260)
    cal.table_bbox = (1, 2, 3, 4)
    assert locate_table(img, cal) == (1, 2, 3, 4)
    cal.table_bbox = None
    with pytest.raises(CalibrationError):
        locate_table(Image.new("RGB", (100, 100), (0, 0, 0)), cal)


def test_sources(tmp_path):
    a = Image.new("RGB", (10, 10), (1, 2, 3))
    a.save(tmp_path / "a.png")
    src = ImageFileSource([tmp_path / "a.png"])
    assert src.capture().getpixel((0, 0)) == (1, 2, 3)
    with pytest.raises(StopIteration):
        src.capture()
    seq = ImageSequenceSource([a, a])
    seq.capture()
    seq.capture()
    with pytest.raises(StopIteration):
        seq.capture()
