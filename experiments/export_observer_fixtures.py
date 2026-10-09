"""Export user-approved frames of a recorded observer session as shareable,
redacted observer fixtures (Phase 44).

    python experiments/export_observer_fixtures.py \\
        --session ~/pokeralpha_sessions/<id> --output /tmp/pokeralpha_fixture_export \\
        [--frames 000012,000031] [--dry-run | --preview] [--margin 0.03] \\
        [--keep-names] [--mask-box X0,Y0,X1,Y1 ...]

What it does, per exported frame:

1. **crop** to one box per session: the union of the located table and every
   calibrated recognition region (+ margin). Browser chrome, other windows
   and the desktop outside it are removed.
2. **mask** the calibrated player-name regions (``seatN_name``) with their
   local background (names are not needed for recognition), plus any
   ``--mask-box`` rectangles (coordinates in the cropped image).
3. **report** text-like content that is inside the crop but outside every
   recognition region (felt logos, chat bubbles, overlays) as
   ``review_needed`` in the manifest and in red in the preview. It is NOT
   masked automatically: blindly masking could destroy UI the recognizers
   need. Inspect it and re-run with ``--mask-box`` if it is personal.
4. copy the frame's annotation (image path rewritten), and write a
   calibration (fixed table box shifted into crop coordinates), a manifest
   with SHA-256 checksums and an export README.

By default only frames whose annotation is ``partial`` or ``complete`` are
exported. Nothing is overwritten (the output directory must be new or
empty), committed or uploaded. ``--dry-run`` writes nothing; ``--preview``
writes only ``preview/*.png`` for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.observer.calibration import TableCalibration, locate_table  # noqa: E402
from poker_alpha.observer.errors import CalibrationError  # noqa: E402
from poker_alpha.observer.session_replay import load_session  # noqa: E402

EXPORT_FORMAT = "pokeralpha.observer_fixture_export/v1"
Box = Tuple[int, int, int, int]


def _union(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def region_boxes(cal: TableCalibration, table: Box) -> Dict[str, Box]:
    return {k: r.to_pixels(table) for k, r in cal.regions.items()}


def crop_box(image_size, cal: TableCalibration, table: Box, margin: float) -> Box:
    boxes = list(region_boxes(cal, table).values()) + [table]
    l, t, r, b = _union(boxes)
    m = int(round(margin * (table[2] - table[0])))
    W, H = image_size
    return (max(0, l - m), max(0, t - m), min(W, r + m), min(H, b + m))


def _shift(box: Box, dx: int, dy: int) -> Box:
    return (box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy)


def paint_background(arr: np.ndarray, box: Box) -> None:
    """Fill ``box`` with the median colour of the pixels just around it."""
    l, t, r, b = box
    H, W = arr.shape[:2]
    l, t, r, b = max(0, l), max(0, t), min(W, r), min(H, b)
    if r <= l or b <= t:
        return
    ring = []
    if t > 0:
        ring.append(arr[t - 1, l:r])
    if b < H:
        ring.append(arr[b, l:r])
    if l > 0:
        ring.append(arr[t:b, l - 1])
    if r < W:
        ring.append(arr[t:b, r])
    colour = np.median(np.concatenate(ring), axis=0) if ring else arr[t:b, l:r].reshape(
        -1, arr.shape[2]).mean(axis=0)
    arr[t:b, l:r] = colour.astype(arr.dtype)


def text_like_boxes(arr: np.ndarray, keep_out: Sequence[Box], min_px: int = 6,
                    max_h: int = 60) -> List[Box]:
    """Small high-contrast blobs grouped into lines, outside ``keep_out``."""
    from scipy import ndimage

    g = arr.astype(np.float32).mean(axis=2)
    bg = ndimage.median_filter(g, size=15)
    mask = np.abs(g - bg) > 60
    for l, t, r, b in keep_out:
        mask[max(0, t - 3):max(0, b + 3), max(0, l - 3):max(0, r + 3)] = False
    lines = ndimage.binary_dilation(mask, structure=np.ones((3, 9)))
    lab, _ = ndimage.label(lines)
    out = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        n = int(mask[sl].sum())
        if n >= min_px and h <= max_h and w >= 4:
            out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return out


def select_frames(data, explicit: Optional[List[str]]) -> List[dict]:
    samples = {s["id"]: s for s in data.samples if s.get("files", {}).get("frame")}
    if explicit:
        missing = [f for f in explicit if f not in samples]
        if missing:
            raise SystemExit(f"frames not in session: {missing}")
        return [samples[f] for f in explicit]
    out = []
    for sid, s in samples.items():
        p = data.path / "annotations" / f"{sid}.json"
        if p.exists() and json.loads(p.read_text()).get("status") in ("partial", "complete"):
            out.append(s)
    return out


def plan_export(session: Path, frames: Optional[List[str]] = None, margin: float = 0.03,
                keep_names: bool = False, mask_boxes: Sequence[Box] = ()) -> dict:
    from PIL import Image

    data = load_session(session)
    cal = data.calibrations[data.initial_checksum]
    chosen = select_frames(data, frames)
    if not chosen:
        raise SystemExit("nothing to export: no frames given and no partial/complete "
                         "annotations in the session")
    tables, sizes = {}, set()
    for s in chosen:
        img = Image.open(data.path / s["files"]["frame"]).convert("RGB")
        sizes.add(img.size)
        try:
            tables[s["id"]] = locate_table(img, cal)
        except CalibrationError as exc:
            raise SystemExit(f"frame {s['id']}: table not found ({exc}); not exported") from exc
    if len(sizes) != 1:
        raise SystemExit(f"frames have different sizes {sorted(sizes)}: export them "
                         "separately (one crop box per export)")
    size = sizes.pop()
    box = _union([crop_box(size, cal, t, margin) for t in tables.values()])
    names = [k for k in cal.regions if k.endswith("_name")]
    warnings = []
    if not keep_names and not names:
        warnings.append("calibration has no seatN_name regions: player names were NOT "
                        "masked; check the preview")
    return {"data": data, "cal": cal, "frames": chosen, "tables": tables, "crop": box,
            "names": [] if keep_names else names, "mask_boxes": list(mask_boxes),
            "warnings": warnings}


def render(plan: dict, sample: dict):
    """Cropped + masked image and the review-needed boxes (crop coordinates)."""
    from PIL import Image

    data, cal, (cl, ct, cr, cb) = plan["data"], plan["cal"], plan["crop"]
    img = Image.open(data.path / sample["files"]["frame"]).convert("RGB")
    table = plan["tables"][sample["id"]]
    regions = region_boxes(cal, table)
    arr = np.asarray(img.crop((cl, ct, cr, cb))).copy()
    masked = []
    for name in plan["names"]:
        b = _shift(regions[name], cl, ct)
        paint_background(arr, b)
        masked.append({"region": name, "box": list(b)})
    for b in plan["mask_boxes"]:
        paint_background(arr, b)
        masked.append({"region": "manual", "box": list(b)})
    keep = [_shift(b, cl, ct) for k, b in regions.items() if not k.endswith("_name")]
    keep += [tuple(m["box"]) for m in masked]
    review = text_like_boxes(arr, keep)
    return Image.fromarray(arr), masked, review, _shift(table, cl, ct)


def export(session: Path, output: Path, frames: Optional[List[str]] = None,
           margin: float = 0.03, keep_names: bool = False,
           mask_boxes: Sequence[Box] = (), dry_run: bool = False,
           preview: bool = False) -> dict:
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()) and not dry_run:
        raise SystemExit(f"{output} exists and is not empty: refusing to overwrite; "
                         "choose a new --output")
    plan = plan_export(Path(session), frames, margin, keep_names, mask_boxes)
    data, cal = plan["data"], plan["cal"]
    cl, ct, cr, cb = plan["crop"]
    manifest = {"format": EXPORT_FORMAT, "source_session": data.session.get("id"),
                "crop_box_in_capture": [cl, ct, cr, cb], "names_masked": not keep_names,
                "warnings": list(plan["warnings"]), "frames": [],
                "privacy": "cropped to table + recognition regions; names masked; "
                           "review_needed boxes were NOT masked - inspect them"}
    if dry_run:
        for s in plan["frames"]:
            manifest["frames"].append({"frame": s["id"], "would_write": f"raw/{_name(data, s)}.png"})
        return manifest
    if preview:
        (output / "preview").mkdir(parents=True, exist_ok=True)
    else:
        for d in ("raw", "annotations"):
            (output / d).mkdir(parents=True, exist_ok=True)
    from PIL import ImageDraw

    for s in plan["frames"]:
        img, masked, review, table = render(plan, s)
        name = _name(data, s)
        entry = {"frame": s["id"], "name": name, "masked": masked,
                 "review_needed": [list(b) for b in review], "table_box": list(table)}
        if preview:
            pv = img.copy()
            d = ImageDraw.Draw(pv)
            for m in masked:
                d.rectangle(m["box"], outline=(120, 120, 255), width=2)
            for b in review:
                d.rectangle(b, outline=(255, 0, 0), width=2)
            pv.save(output / "preview" / f"{name}.png")
        else:
            path = output / "raw" / f"{name}.png"
            img.save(path, format="PNG")
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            ann_p = data.path / "annotations" / f"{s['id']}.json"
            if ann_p.exists():
                ann = json.loads(ann_p.read_text())
                ann["image"] = f"raw/{name}.png"
                ann.pop("prefilled_from_observer", None)
                ann["source"] = {"session": data.session.get("id"), "frame": s["id"],
                                 "crop_box_in_capture": [cl, ct, cr, cb]}
                (output / "annotations" / f"{name}.json").write_text(json.dumps(ann, indent=1))
                entry["annotation"] = {"status": ann.get("status"), "role": ann.get("role")}
                if ann.get("role", "unassigned") == "unassigned":
                    manifest["warnings"].append(
                        f"{name}: role unassigned; set validation / held_out before using "
                        "it for accuracy")
        manifest["frames"].append(entry)
    if not preview:
        out_cal = TableCalibration.from_dict(cal.to_dict())
        if out_cal.table_bbox is not None:
            out_cal.table_bbox = _shift(tuple(out_cal.table_bbox), cl, ct)
        out_cal.save(output / "calibration.json")
        (output / "README.txt").write_text(
            "PokerAlpha observer fixture export (local; never uploaded automatically).\n"
            "Before sharing: open every raw/*.png and check the review_needed boxes in\n"
            "manifest.json (text outside recognition regions that was NOT masked).\n"
            "Score with: python experiments/observer_validation.py --fixture-dir <this dir>\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _name(data, sample) -> str:
    return f"{data.session.get('id', 'session')}_{sample['id']}"


def _box(text: str) -> Box:
    v = [int(x) for x in text.split(",")]
    if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
        raise argparse.ArgumentTypeError("box must be X0,Y0,X1,Y1 with X1>X0, Y1>Y0")
    return tuple(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", default=None, help="comma-separated frame ids")
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--keep-names", action="store_true")
    p.add_argument("--mask-box", type=_box, action="append", default=[])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--preview", action="store_true")
    a = p.parse_args(argv)
    m = export(a.session, a.output, a.frames.split(",") if a.frames else None, a.margin,
               a.keep_names, a.mask_box, a.dry_run, a.preview)
    kind = "dry run (nothing written)" if a.dry_run else ("preview" if a.preview else "export")
    print(f"{kind}: {len(m['frames'])} frame(s), crop {m['crop_box_in_capture']}")
    for f in m["frames"]:
        rv = f.get("review_needed", [])
        print(f"  {f['frame']}: masked {len(f.get('masked', []))} region(s), "
              f"{len(rv)} text-like area(s) left for review")
    for w in m["warnings"]:
        print("WARNING:", w)
    if not a.dry_run:
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
