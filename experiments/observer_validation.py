"""Experiment: screen-observer validation on SYNTHETIC frames.

Measures per-frame card / numeric / discrete-field / whole-state accuracy of
the PokerNow-style adapter, the mean confidence of correct vs wrong readings,
the state accuracy after fusing 3 noisy frames with the StateTracker, and
the false event rate (events inferred while watching a table that does not
change).

These numbers describe the synthetic renderer only. Exact PokerNow visual
accuracy is not validated without representative screenshots.

Usage
-----
    python experiments/observer_validation.py --frames 30 --seed 0
    python experiments/observer_validation.py --fixture-dir tests/fixtures/pokernow

With ``--fixture-dir`` it instead scores annotated screenshots (see
``tests/fixtures/pokernow/README.md``) field by field and writes
``results/validation/observer_fixture_validation.json``.

Outputs (under --outdir, default ./results):
    data/observer_synthetic_validation.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from poker_alpha.observer.evaluation import evaluate, frame_errors
from poker_alpha.observer.fusion import StateTracker
from poker_alpha.observer.pokernow import PokerNowStyleAdapter, default_layout
from poker_alpha.observer.synthetic import random_table, render_table

CONFIGS = [  # seats, size, pixel-noise std
    (2, (1280, 800), 0.0), (3, (1280, 800), 0.0), (6, (1280, 800), 0.0),
    (6, (1280, 800), 8.0), (6, (1920, 1200), 0.0), (9, (1280, 800), 0.0),
    (9, (1600, 1000), 4.0), (9, (960, 600), 0.0),
]


def fused_state_ok(adapter, cal, table, seed: int, noise: float, size) -> bool:
    tr = StateTracker(cal, 0.5, 1.0)
    for k in range(3):
        img = render_table(table, cal, size, noise=max(noise, 3.0), seed=seed * 10 + k)
        tr.update(adapter.read_frame(img))
    snap = tr.snapshot()
    ok = snap.board == tuple(table.board) and snap.hero_cards == tuple(table.hero_cards)
    ok &= snap.pot == table.pot and snap.dealer == table.dealer
    for s, seat in enumerate(table.seats):
        if seat.stack is None:
            ok &= not snap.occupied[s]
            continue
        ok &= snap.stacks[s] == (0.0 if seat.all_in else seat.stack)
        ok &= snap.bets[s] == seat.bet
    return bool(ok)


def false_events(adapter, cal, table, frames: int, noise: float, size, seed: int) -> int:
    tr = StateTracker(cal, 0.5, 1.0)
    for k in range(frames):
        img = render_table(table, cal, size, noise=max(noise, 3.0), seed=seed + k)
        events = tr.update(adapter.read_frame(img))
        if k >= 3:  # after the initial confirmation window nothing may happen
            yield len(events)


def fixture_mode(fixture_dir: Path, out_path: Path) -> dict:
    import json

    from PIL import Image

    from poker_alpha.observer.annotations import load_fixture_dir, score
    from poker_alpha.observer.calibration import TableCalibration

    anns = load_fixture_dir(fixture_dir)
    cal_file = fixture_dir / "calibration.json"
    cal_fixed = TableCalibration.load(cal_file) if cal_file.exists() else None
    frames, missing = [], []
    for a in anns:
        if not a.image.exists():
            missing.append(str(a.image))
            continue
        cal = cal_fixed or default_layout(a.num_seats, a.hero_seat)
        if cal.num_seats != a.num_seats:
            raise SystemExit(f"{a.name}: calibration has {cal.num_seats} seats, "
                             f"annotation {a.num_seats}")
        obs = PokerNowStyleAdapter(cal).read_frame(Image.open(a.image).convert("RGB"))
        frames.append((obs, a))
    result = {"fixture_dir": str(fixture_dir),
              "calibration": str(cal_file) if cal_fixed else "default_layout (not PokerNow-derived)",
              "annotations": len(anns), "missing_images": missing,
              "metrics": score(frames)}
    if not frames:
        result["note"] = ("no annotated screenshots found: nothing was measured; "
                          "real PokerNow accuracy remains unvalidated")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    m = result["metrics"]
    print(f"annotated screenshots scored: {m['screenshots']} (of {len(anns)} annotations)")
    if missing:
        print(f"missing images: {missing}")
    for k in ("hero_cards", "board_cards", "stack", "bet", "pot", "dealer",
              "seat_occupancy", "full_state"):
        v = m[k]
        acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.3f}"
        mae = "" if v.get("mae") is None else f"  MAE {v['mae']:.3f}"
        unr = "" if not v.get("unreadable") else f"  unreadable {v['unreadable']}"
        print(f"  {k:<15} {acc:>6}  ({v['correct']}/{v['total']}){mae}{unr}")
    if not frames:
        print(result["note"])
    print(f"wrote {out_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--fixture-dir", type=Path, default=None)
    p.add_argument("--out", type=Path,
                   default=Path("results/validation/observer_fixture_validation.json"))
    args = p.parse_args()
    if args.fixture_dir is not None:
        fixture_mode(args.fixture_dir, args.out)
        return
    rows = []
    for n, size, noise in CONFIGS:
        rng = np.random.default_rng(args.seed + n)
        cal = default_layout(n, 0)
        ad = PokerNowStyleAdapter(cal)
        tables = [random_table(rng, n) for _ in range(args.frames)]
        start = time.perf_counter()
        frames = [(render_table(t, cal, size, noise=noise, seed=i), t)
                  for i, t in enumerate(tables)]
        rep = evaluate(ad, frames)
        per_frame = (time.perf_counter() - start) / len(frames)
        fused = np.mean([fused_state_ok(ad, cal, t, i, noise, size)
                         for i, t in enumerate(tables[:10])])
        fe = list(false_events(ad, cal, tables[0], 12, noise, size, args.seed))
        rows.append({
            "seats": n, "width": size[0], "height": size[1], "noise_std": noise,
            "frames": rep.frames, "card_accuracy": rep.card_accuracy,
            "numeric_accuracy": rep.numeric_accuracy,
            "discrete_accuracy": rep.discrete_accuracy,
            "state_accuracy": rep.state_accuracy,
            "fused_state_accuracy_3frames": fused,
            "mean_conf_correct": rep.mean_conf_correct,
            "mean_conf_wrong": rep.mean_conf_wrong,
            "false_events_per_frame": sum(fe) / len(fe),
            "seconds_per_frame_incl_render": per_frame,
        })
        print(rows[-1])
    out = args.outdir / "data"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "observer_synthetic_validation.csv", index=False)
    print(df.to_string(index=False))
    print("\nSYNTHETIC frames only: exact PokerNow visual accuracy is not "
          "validated without representative screenshots.")


if __name__ == "__main__":
    main()
