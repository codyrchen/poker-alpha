"""Observer robustness matrix (Phase 49) — SYNTHETIC / DERIVED ROBUSTNESS ONLY.

Deterministic perturbations of
  * synthetic rendered tables (exact ground truth), and
  * the single real PokerNow TUNING frame (tests/fixtures/pokernow), every
    row of which is labelled "DERIVED FROM TUNING FRAME — NOT INDEPENDENT
    VALIDATION".
Nothing here measures real PokerNow accuracy; it measures how the pipeline
degrades as the picture degrades.

Per perturbation level:
  table_found        fraction of frames whose table was located
  alignment_iou      IoU of the located table box with the expected one
  cards_correct      fraction of card fields equal to the truth
  amounts_correct    fraction of pot / stack / bet fields equal to the truth
  conf_delta         mean confidence change vs the clean frame (card + amount fields)
  recovery_frames    (synthetic) clean frames needed, after 3 perturbed frames, until
                     live_check passes again (None = never within 10)
  state_changed      (synthetic) a confirmed critical value changed during the burst

    python experiments/observer_robustness.py [--out results/validation/observer_robustness_v1.json]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.observer.calibration import TableCalibration  # noqa: E402
from poker_alpha.observer.errors import CalibrationError  # noqa: E402
from poker_alpha.observer.live import (CaptureSettings, LiveObserverSession,  # noqa: E402
                                       live_check)
from poker_alpha.observer.pokernow import (PokerNowStyleAdapter, default_layout,  # noqa: E402
                                           )
from poker_alpha.observer.synthetic import random_table, render_table  # noqa: E402

LABEL = "SYNTHETIC / DERIVED ROBUSTNESS ONLY - not real PokerNow accuracy"
DERIVED = "DERIVED FROM TUNING FRAME — NOT INDEPENDENT VALIDATION"
CARD_FIELDS = ["hero_card_0", "hero_card_1"] + [f"board_{i}" for i in range(5)]


# -- perturbations: (name, level, fn(img) -> (img, box_map)) ------------------------
# box_map maps the clean table box to the expected box in the perturbed image.

def _same(b):
    return b


def _scale_map(f):
    return lambda b: tuple(int(round(v * f)) for v in b)


def _shift_map(dx, dy):
    return lambda b: (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def noise(img, sd, seed=0):
    a = np.asarray(img).astype(float)
    a += np.random.default_rng(seed).normal(0, sd, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def pad(img, dx, dy, colour=(36, 36, 42)):
    out = Image.new("RGB", (img.width + abs(dx) * 2, img.height + abs(dy) * 2), colour)
    out.paste(img, (abs(dx) + dx, abs(dy) + dy))
    return out


def rotate(img, deg):
    return img.rotate(deg, resample=Image.BICUBIC, fillcolor=(36, 36, 42))


def zoom(img, f):
    """Browser zoom approximation: content scaled, viewport size kept."""
    z = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
    canvas = Image.new("RGB", img.size, (36, 36, 42))
    canvas.paste(z, ((img.width - z.width) // 2, (img.height - z.height) // 2))
    return canvas


def perturbations():
    P = []
    for f in (0.5, 0.67, 0.8, 1.25, 1.5, 2.0):
        P.append(("scale", f, lambda im, f=f: (im.resize((int(im.width * f), int(im.height * f)),
                                                         Image.LANCZOS), _scale_map(f))))
    for f in (1.25, 1.5, 0.7, 0.5):
        P.append(("brightness", f, lambda im, f=f: (ImageEnhance.Brightness(im).enhance(f), _same)))
    for f in (0.7, 0.5, 1.3):
        P.append(("contrast", f, lambda im, f=f: (ImageEnhance.Contrast(im).enhance(f), _same)))
    for r in (0.5, 1.0, 1.5, 2.0):
        P.append(("blur", r, lambda im, r=r: (im.filter(ImageFilter.GaussianBlur(r)), _same)))
    for sd in (4, 8, 16):
        P.append(("gaussian_noise", sd, lambda im, sd=sd: (noise(im, sd), _same)))
    for q in (80, 50, 30, 15):
        P.append(("jpeg_quality", q, lambda im, q=q: (jpeg(im, q), _same)))
    for d in (5, 25, 80):
        P.append(("crop_offset", d, lambda im, d=d: (im.crop((d, d, im.width, im.height)),
                                                    _shift_map(-d, -d))))
    for deg in (0.5, 1.0, 2.0):
        P.append(("rotation_deg", deg, lambda im, deg=deg: (rotate(im, deg), _same)))
    for d in (40, 150):
        P.append(("translation", d, lambda im, d=d: (pad(im, d, d // 2),
                                                    _shift_map(2 * d, d))))
    for f in (0.9, 0.8, 1.1):
        def _zoom(im, f=f):
            w, h = im.size
            zw, zh = int(w * f), int(h * f)
            ox, oy = (w - zw) // 2, (h - zh) // 2
            return zoom(im, f), (lambda b: (int(b[0] * f) + ox, int(b[1] * f) + oy,
                                             int(b[2] * f) + ox, int(b[3] * f) + oy))
        P.append(("browser_zoom", f, _zoom))
    return P


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def truth_fields(table, cal):
    t = {"pot": float(table.pot)}
    for i in range(2):
        t[f"hero_card_{i}"] = table.hero_cards[i]
    for i in range(5):
        t[f"board_{i}"] = table.board[i] if i < len(table.board) else None
    for s, seat in enumerate(table.seats):
        if seat.stack is None:
            continue
        t[f"seat{s}.stack"] = 0.0 if seat.all_in else float(seat.stack)
        t[f"seat{s}.bet"] = float(seat.bet)
    return t


def score_frame(adapter, cal, img, truth, clean_box, box_map, clean_conf):
    try:
        obs = adapter.read_frame(img)
    except CalibrationError:
        return {"found": False}
    want = box_map(clean_box)
    cards = [obs.value(k) == truth[k] for k in CARD_FIELDS if k in truth]
    amt_keys = [k for k in truth if k not in CARD_FIELDS]
    amounts = [obs.value(k) == truth[k] for k in amt_keys]
    deltas = [obs.confidence(k) - clean_conf[k] for k in truth if k in clean_conf]
    return {"found": True, "iou": iou(obs.table_bbox, want),
            "cards": sum(cards) / len(cards) if cards else None,
            "amounts": sum(amounts) / len(amounts) if amounts else None,
            "conf_delta": float(np.mean(deltas)) if deltas else None}


def recovery(cal, clean, bad):
    frames = [clean] * 3 + [bad] * 3 + [clean] * 10

    class Scr:
        i = 0

        def capture(self):
            f = frames[min(self.i, len(frames) - 1)]
            self.i += 1
            return f
    scr = Scr()
    s = LiveObserverSession(source_factory=lambda m, r: scr)
    s.configure(CaptureSettings({"left": 0, "top": 0, "width": 0, "height": 0}, None),
                cal, 0.5, 1.0)
    for i in range(3):
        s.step(now=float(i))
    keys = ["hero_card_0", "hero_card_1", "pot"] + [k for k in s.tracker.fields
                                                     if k.endswith(".stack")]
    before = {k: s.tracker._stable(k) for k in keys}
    for i in range(3):
        s.step(now=3.0 + i)
    changed = {k: s.tracker._stable(k) for k in keys} != before
    for n in range(1, 11):
        s.step(now=6.0 + n)
        if live_check(s, 0.0).ok:
            return n, changed
    return None, changed


def run(out: Path, n_tables: int = 8, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    synth = []
    for k in range(n_tables):
        cal = default_layout([2, 6, 9][k % 3], 0)
        tbl = random_table(rng, cal.num_seats)
        synth.append((cal, tbl, render_table(tbl, cal)))
    fixture = ROOT / "tests" / "fixtures" / "pokernow"
    real = json.loads((fixture / "annotations" / "hu_preflop_0001.json").read_text())
    real_img = Image.open(fixture / "raw" / "hu_preflop_0001.png").convert("RGB")
    real_cal = TableCalibration.load(fixture / "calibration.json")
    real_truth = {"hero_card_0": real["hero_cards"][0], "hero_card_1": real["hero_cards"][1],
                  "pot": real["pot"], **{f"board_{i}": None for i in range(5)}}
    for s in real["seats"]:
        real_truth[f"seat{s['seat']}.stack"] = s["stack"]
        real_truth[f"seat{s['seat']}.bet"] = s["bet"]
    rows = []
    for name, level, fn in perturbations():
        agg = {"synthetic": [], "derived": []}
        rec = []
        for cal, tbl, img in synth:
            ad = PokerNowStyleAdapter(cal)
            clean = ad.read_frame(img)
            truth = truth_fields(tbl, cal)
            conf = {k: clean.confidence(k) for k in truth}
            bad, box_map = fn(img)
            agg["synthetic"].append(score_frame(ad, cal, bad, truth, clean.table_bbox,
                                                box_map, conf))
            if cal.num_seats == 2:
                rec.append(recovery(cal, img, bad))
        ad = PokerNowStyleAdapter(real_cal)
        clean = ad.read_frame(real_img)
        conf = {k: clean.confidence(k) for k in real_truth}
        bad, box_map = fn(real_img)
        agg["derived"].append(score_frame(ad, real_cal, bad, real_truth, clean.table_bbox,
                                          box_map, conf))
        for src, frames in agg.items():
            found = [f for f in frames if f["found"]]

            def mean(k):
                v = [f[k] for f in found if f.get(k) is not None]
                return round(float(np.mean(v)), 4) if v else None
            row = {"perturbation": name, "level": level, "source": src,
                   "frames": len(frames), "table_found": len(found) / len(frames),
                   "alignment_iou": mean("iou"), "cards_correct": mean("cards"),
                   "amounts_correct": mean("amounts"), "conf_delta": mean("conf_delta")}
            if src == "synthetic":
                rf = [r[0] for r in rec]
                row["recovery_frames"] = (max(rf) if all(r is not None for r in rf) else None)
                row["state_changed_during_burst"] = any(r[1] for r in rec)
            else:
                row["label"] = DERIVED
            rows.append(row)
    result = {"label": LABEL, "derived_label": DERIVED, "seed": seed,
              "synthetic_tables": n_tables, "rows": rows,
              "seconds": round(time.perf_counter() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "validation" / "observer_robustness_v1.json")
    p.add_argument("--tables", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    res = run(a.out, a.tables, a.seed)
    print(LABEL)
    print(f"{'perturbation':<15}{'level':>7}  {'source':<9}{'found':>6}{'IoU':>7}{'cards':>7}"
          f"{'amounts':>8}{'dconf':>7}{'recov':>6}{'chg':>5}")
    for r in res["rows"]:
        def f(v, fmt="{:.2f}"):
            return "  -" if v is None else fmt.format(v)
        print(f"{r['perturbation']:<15}{r['level']:>7}  {r['source']:<9}{r['table_found']:>6.2f}"
              f"{f(r['alignment_iou']):>7}{f(r['cards_correct']):>7}{f(r['amounts_correct']):>8}"
              f"{f(r['conf_delta'], '{:+.2f}'):>7}"
              f"{f(r.get('recovery_frames'), '{}'):>6}"
              f"{('yes' if r.get('state_changed_during_burst') else ''):>5}")
    print(f"derived rows: {DERIVED}")
    print(f"wrote {a.out} in {res['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
