"""Screenshot-fixture demo: image -> recognized state -> (optional) decision.

    python -m poker_alpha.observer.demo tests/fixtures/table.png
        [--calibration CAL.json] [--frames 3] [--sb 0.5] [--bb 1.0]
        [--no-decide] [--rollouts N]

Prints every recognized field with its confidence, the fused table state,
validation warnings and — unless ``--no-decide`` — the DecisionReport.

Exact PokerNow visual accuracy is not validated without representative
screenshots; the bundled fixture is synthetic. Read-only: nothing is clicked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    from PIL import Image

    from ..decision import DecisionConfig, recommend_action
    from ..holdem import validate
    from .calibration import TableCalibration
    from .fusion import StateTracker
    from .pokernow import PokerNowStyleAdapter, default_layout

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--frames", type=int, default=3,
                   help="times to feed the image (fusion needs agreement)")
    p.add_argument("--sb", type=float, default=0.5)
    p.add_argument("--bb", type=float, default=1.0)
    p.add_argument("--seats", type=int, default=6)
    p.add_argument("--no-decide", action="store_true")
    p.add_argument("--rollouts", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cal_path = args.calibration
    if cal_path is None:
        sibling = args.image.with_name("table_calibration.json")
        cal_path = sibling if sibling.exists() else None
    cal = TableCalibration.load(cal_path) if cal_path else default_layout(args.seats)
    adapter = PokerNowStyleAdapter(cal)
    image = Image.open(args.image).convert("RGB")
    print(f"calibration: {cal.name} ({'file ' + str(cal_path) if cal_path else 'default layout'})")

    frame = adapter.read_frame(image)
    print("\n=== Recognized fields (value, confidence) ===")
    for name, f in frame.fields.items():
        if name.endswith(".all_in") and not f.value:
            continue  # only show all-in flags that are set
        print(f"  {name:<18} {str(f.value):<10} {f.confidence:5.2f}")

    tracker = StateTracker(cal, args.sb, args.bb)
    tracker.update(frame)
    for _ in range(max(args.frames - 1, 0)):
        tracker.update(adapter.read_frame(image))
    tracked = tracker.tracked()
    obs = tracker.to_observed_state()
    print("\n=== Fused state ===")
    print(f"  {obs.describe() if obs.hero_cards else 'hero cards not confirmed'}")
    print(f"  critical field confidence: {tracker.critical_confidence():.2f}")
    for w in tracked.warnings:
        print(f"  tracker: {w}")
    print("\n=== Validation ===")
    issues = validate(obs)
    if not issues:
        print("  ok")
    for i in issues:
        print(f"  {i.severity}: {i.message}")
    if not args.no_decide:
        print("\n=== Decision ===")
        report = recommend_action(obs, config=DecisionConfig(
            equity_simulations=2000, rollout_simulations=args.rollouts,
            seed=args.seed, observer_confidence=tracker.critical_confidence()))
        print(report.format())
    print("\nSynthetic-fixture demo: exact PokerNow visual accuracy is not "
          "validated without representative screenshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
