"""Phase 44: privacy-preserving fixture export (synthetic session).

Synthetic frames are pasted into a larger 'desktop' with fake browser chrome
and a fake name / chat, so cropping and masking can be checked."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")

from PIL import Image, ImageDraw  # noqa: E402

from observer_helpers import CAL, record, table  # noqa: E402

from poker_alpha.observer.annotation_tool import (SessionAnnotator,  # noqa: E402
                                                  build_annotation)
from poker_alpha.observer.synthetic import render_table  # noqa: E402

pytestmark = pytest.mark.vision
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from export_observer_fixtures import export, main  # noqa: E402

DESK = (1600, 1100)
OFFSET = (160, 140)
CHROME = (0, 0, 1600, 60)          # "browser tabs + URL" strip
SECRET = (1400, 1000)              # private text far outside the table


def desktop(t):
    img = Image.new("RGB", DESK, (90, 60, 120))
    d = ImageDraw.Draw(img)
    d.rectangle(CHROME, fill=(230, 230, 230))
    d.text((20, 20), "inbox (6,438) - private tab", fill=(0, 0, 0))
    d.text(SECRET, "PRIVATE CHAT", fill=(255, 255, 255))
    img.paste(render_table(t, CAL), OFFSET)
    return img


@pytest.fixture
def annotated_session(tmp_path):
    tables = [table(), table(board=("Qs", "Jh", "4c"), pot=3.0)]
    frames = [desktop(tables[0])] * 4 + [desktop(tables[1])] * 4
    rec, s = record(tmp_path / "sessions", frames)
    rec.finish()
    tool = SessionAnnotator(rec.path)
    sid = tool.ids[-1]
    tool.save(sid, build_annotation(tool.empty(sid), status="complete", role="validation",
                                    hero_cards=("As", "Kd"), board="Qs Jh 4c", pot="3"))
    return rec.path, sid


def test_dry_run_writes_nothing(annotated_session, tmp_path):
    path, sid = annotated_session
    out = tmp_path / "export"
    m = export(path, out, dry_run=True)
    assert [f["frame"] for f in m["frames"]] == [sid]
    assert not out.exists()


def test_export_crops_masks_and_copies_annotations(annotated_session, tmp_path):
    path, sid = annotated_session
    out = tmp_path / "export"
    m = export(path, out)
    l, t, r, b = m["crop_box_in_capture"]
    assert l >= OFFSET[0] - 60 and t > CHROME[3]               # browser chrome gone
    assert r < SECRET[0] or b < SECRET[1]                      # private text outside crop
    f = m["frames"][0]
    img = Image.open(out / "raw" / f"{f['name']}.png")
    assert img.size == (r - l, b - t)
    assert {x["region"] for x in f["masked"]} == {"seat0_name", "seat1_name"}
    # a masked name box is now (almost) uniform
    x0, y0, x1, y1 = f["masked"][0]["box"]
    assert np.asarray(img.convert("L"))[y0:y1, x0:x1].std() < 8
    ann = json.loads((out / "annotations" / f"{f['name']}.json").read_text())
    assert ann["image"] == f"raw/{f['name']}.png" and ann["role"] == "validation"
    assert ann["source"]["frame"] == sid
    assert (out / "calibration.json").exists() and (out / "README.txt").exists()
    assert len(f["sha256"]) == 64
    # the export is a valid fixture directory for the validation harness
    from observer_validation import fixture_mode

    res = fixture_mode(out, tmp_path / "r.json")
    assert res["real_validation"]["status"] == "MEASURED"
    assert res["real_validation"]["metrics"]["hero_exact_pair"]["accuracy"] == 1.0


def test_never_overwrites(annotated_session, tmp_path):
    path, _ = annotated_session
    out = tmp_path / "export"
    export(path, out)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        export(path, out)


def test_preview_and_unmasked_text_is_reported_not_masked(annotated_session, tmp_path):
    path, sid = annotated_session
    out = tmp_path / "pv"
    assert main(["--session", str(path), "--output", str(out), "--preview"]) == 0
    assert list((out / "preview").glob("*.png")) and not (out / "raw").exists()
    m = json.loads((out / "manifest.json").read_text())
    # the synthetic renderer draws no text outside recognition regions, so nothing
    # is flagged; a manual --mask-box is applied and recorded
    out2 = tmp_path / "e2"
    assert main(["--session", str(path), "--output", str(out2),
                 "--mask-box", "0,0,20,20"]) == 0
    m2 = json.loads((out2 / "manifest.json").read_text())
    assert any(x["region"] == "manual" for x in m2["frames"][0]["masked"])
    assert isinstance(m["frames"][0]["review_needed"], list)


def test_text_outside_regions_is_flagged(tmp_path):
    from export_observer_fixtures import text_like_boxes

    arr = np.full((200, 300, 3), 40, np.uint8)
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).text((150, 150), "hello chat", fill=(255, 255, 255))
    ImageDraw.Draw(img).text((20, 20), "49.75", fill=(255, 255, 255))
    boxes = text_like_boxes(np.asarray(img), keep_out=[(10, 10, 80, 40)])
    assert len(boxes) == 1 and boxes[0][0] >= 140
