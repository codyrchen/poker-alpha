"""Observer dataset infrastructure: manifests, session-level splits,
leakage checks, coverage reports and the baseline recognizer benchmark.

    python -m poker_alpha.observer.dataset manifest  <root>
    python -m poker_alpha.observer.dataset split     <root> [--seed N]
                                                     [--fractions 0.6,0.2,0.2]
                                                     [--lock]
    python -m poker_alpha.observer.dataset check     <root>
    python -m poker_alpha.observer.dataset coverage  <root>
    python -m poker_alpha.observer.dataset baseline  <root> [--out report.json]

A dataset root contains one directory per RECORDED SESSION, each with
``annotations/*.json`` (``pokeralpha.screenshot_annotation/v1``) and the
referenced images. Splits are assigned PER SESSION — adjacent frames of one
session are near-duplicates, so frame-level splits would leak; the check
command verifies this and also catches byte-identical images appearing in
different splits.

Once a held-out assignment exists it can be LOCKED (``split --lock``):
``heldout.lock.json`` records the held-out session set, and later split
runs refuse to move those sessions. Touching held-out data before the final
evaluation invalidates it — the tooling makes that loud, not convenient.

NOTHING here fabricates data: with no annotated sessions in a split, the
baseline reports that split as BLOCKED instead of inventing numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .annotations import (Annotation, load_annotation, load_splits,
                          validate_annotation)

DATASET_FORMAT = "pokeralpha.observer_dataset/v1"
SPLIT_ROLES = ("tuning", "validation", "held_out")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_dirs(root: Path) -> List[Path]:
    """Direct children that look like recorded sessions."""
    out = []
    for p in sorted(Path(root).iterdir()):
        if p.is_dir() and (p / "annotations").is_dir():
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def build_manifest(root: Path) -> dict:
    sessions = []
    problems: List[str] = []
    for sd in session_dirs(root):
        frames = []
        for ap in sorted((sd / "annotations").glob("*.json")):
            d = json.loads(ap.read_text())
            errs = validate_annotation(d)
            if errs:
                problems.append(f"{ap}: {'; '.join(errs)}")
                continue
            img = sd / d["image"]
            frames.append({
                "name": ap.stem,
                "annotation_sha256": _sha(ap),
                "image": d["image"],
                "image_sha256": _sha(img) if img.exists() else None,
                "status": d.get("status", "complete"),
                "role": d.get("role", "unassigned"),
            })
            if not img.exists():
                problems.append(f"{ap}: image {d['image']} missing")
        sessions.append({"name": sd.name, "path": str(sd),
                         "frames": frames, "n_frames": len(frames)})
    roles = load_root_splits(root)
    return {
        "format": DATASET_FORMAT,
        "root": str(root),
        "sessions": sessions,
        "session_roles": roles,
        "counts": {
            "sessions": len(sessions),
            "frames": sum(s["n_frames"] for s in sessions),
        },
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# session-level splits
# ---------------------------------------------------------------------------

def load_root_splits(root: Path) -> Dict[str, str]:
    p = Path(root) / "splits.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_lock(root: Path) -> Optional[dict]:
    p = Path(root) / "heldout.lock.json"
    return json.loads(p.read_text()) if p.exists() else None


def assign_splits(root: Path, seed: int = 0,
                  fractions=(0.6, 0.2, 0.2), lock: bool = False) -> dict:
    """Deterministic SESSION-level split. Locked held-out sessions never
    move; new sessions fill the quotas around them."""
    if abs(sum(fractions) - 1.0) > 1e-9 or len(fractions) != 3:
        raise ValueError("fractions must be three numbers summing to 1")
    sessions = [p.name for p in session_dirs(root)]
    if not sessions:
        raise SystemExit(f"no sessions under {root}")
    existing_lock = load_lock(root)
    locked_held = set(existing_lock["held_out_sessions"]) if existing_lock else set()
    missing = locked_held - set(sessions)
    if missing:
        raise SystemExit(f"locked held-out sessions missing from disk: "
                         f"{sorted(missing)} — refusing to continue")
    rng = random.Random(seed)
    movable = sorted(set(sessions) - locked_held)
    rng.shuffle(movable)
    n = len(sessions)
    want_held = max(round(fractions[2] * n), 1 if n >= 3 else 0)
    want_val = max(round(fractions[1] * n), 1 if n >= 3 else 0)
    held = list(locked_held)
    while len(held) < want_held and movable:
        held.append(movable.pop())
    val = [movable.pop() for _ in range(min(want_val, len(movable)))]
    assign = {s: "tuning" for s in movable}
    assign.update({s: "validation" for s in val})
    assign.update({s: "held_out" for s in held})
    (Path(root) / "splits.json").write_text(json.dumps(assign, indent=1,
                                                       sort_keys=True))
    # mirror into each session dir so annotation loading enforces the role
    for sd in session_dirs(root):
        role = assign[sd.name]
        stems = sorted(p.stem for p in (sd / "annotations").glob("*.json"))
        (sd / "splits.json").write_text(
            json.dumps({s: role for s in stems}, indent=1, sort_keys=True))
    out = {"assignments": assign, "seed": seed, "fractions": list(fractions),
           "locked": sorted(locked_held)}
    if lock:
        lock_doc = {
            "format": DATASET_FORMAT,
            "held_out_sessions": sorted(s for s, r in assign.items()
                                        if r == "held_out"),
            "note": ("LOCKED: these sessions are reserved for the final "
                     "held-out evaluation. Do not tune, debug or even look "
                     "at them. Deleting this file invalidates every "
                     "held-out claim made after it was created."),
        }
        (Path(root) / "heldout.lock.json").write_text(
            json.dumps(lock_doc, indent=1))
        out["lock"] = lock_doc
    return out


def check_splits(root: Path) -> List[str]:
    """Leakage checks. Empty list = clean."""
    errs: List[str] = []
    assign = load_root_splits(root)
    dirs = session_dirs(root)
    if not assign:
        return [f"no splits.json under {root} — run split first"]
    for sd in dirs:
        if sd.name not in assign:
            errs.append(f"session {sd.name} has no split assignment")
            continue
        role = assign[sd.name]
        per = load_splits(sd)
        stems = {p.stem for p in (sd / "annotations").glob("*.json")}
        for stem in stems:
            r = per.get(stem)
            if r is None:
                errs.append(f"{sd.name}/{stem}: not in session splits.json")
            elif r != role:
                errs.append(f"{sd.name}/{stem}: role {r} != session role "
                            f"{role} (a session must live in ONE split)")
    # byte-identical images in different splits = frame leakage via copies
    by_sha: Dict[str, set] = defaultdict(set)
    for sd in dirs:
        for ap in (sd / "annotations").glob("*.json"):
            d = json.loads(ap.read_text())
            img = sd / d.get("image", "")
            if img.exists():
                by_sha[_sha(img)].add(assign.get(sd.name, "?"))
    for sha, roles in by_sha.items():
        if len(roles) > 1:
            errs.append(f"image {sha[:12]}… appears in multiple splits "
                        f"{sorted(roles)}")
    lock = load_lock(root)
    if lock:
        now_held = {s for s, r in assign.items() if r == "held_out"}
        for s in lock["held_out_sessions"]:
            if s not in now_held:
                errs.append(f"LOCK VIOLATION: session {s} left held_out "
                            f"after locking")
    return errs


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def _bucket(x: Optional[float], edges) -> str:
    if x is None:
        return "unknown"
    for e in edges:
        if x <= e:
            return f"<= {e:g}"
    return f"> {edges[-1]:g}"


def coverage_report(root: Path) -> dict:
    ranks: Counter = Counter()
    suits: Counter = Counter()
    streets: Counter = Counter()
    pots: Counter = Counter()
    stacks: Counter = Counter()
    bets: Counter = Counter()
    dealers: Counter = Counter()
    hero_seats: Counter = Counter()
    folded = 0
    allin = 0
    resolutions: Counter = Counter()
    frames = 0
    for sd in session_dirs(root):
        for ann in load_fixture_annotations(sd):
            frames += 1
            for c in (ann.hero_cards or ()) + tuple(ann.board or ()):
                if c:
                    ranks[c[0]] += 1
                    suits[c[1]] += 1
            streets[ann.street or "unknown"] += 1
            pots[_bucket(ann.pot, (2, 5, 10, 25, 50, 100))] += 1
            for s in ann.seats.values():
                if s.stack is not None:
                    stacks[_bucket(s.stack, (10, 25, 50, 100, 200))] += 1
                if s.bet:
                    bets[_bucket(s.bet, (1, 2.5, 5, 10, 25))] += 1
                if s.folded:
                    folded += 1
                if getattr(s, "all_in", None):
                    allin += 1
            if ann.dealer_seat is not None:
                dealers[ann.dealer_seat] += 1
            hero_seats[ann.hero_seat] += 1
            try:
                from PIL import Image

                with Image.open(ann.image) as im:
                    resolutions[f"{im.width}x{im.height}"] += 1
            except Exception:
                resolutions["unreadable"] += 1
    all_ranks = set("AKQJT98765432")
    all_suits = set("shdc")
    missing = sorted(all_ranks - set(ranks)) + sorted(all_suits - set(suits))
    missing_streets = sorted({"preflop", "flop", "turn", "river"} - set(streets))
    return {
        "format": DATASET_FORMAT, "frames": frames,
        "card_ranks": dict(ranks), "card_suits": dict(suits),
        "streets": dict(streets), "pot_buckets": dict(pots),
        "stack_buckets": dict(stacks), "bet_buckets": dict(bets),
        "dealer_seats": {str(k): v for k, v in dealers.items()},
        "hero_seats": {str(k): v for k, v in hero_seats.items()},
        "folded_seat_observations": folded,
        "all_in_seat_observations": allin,
        "resolutions": dict(resolutions),
        "missing_coverage": {
            "card_symbols_never_seen": missing,
            "streets_never_seen": missing_streets,
        },
        "note": ("coverage of the LABELED data only; it says nothing about "
                 "recognition accuracy"),
    }


def load_fixture_annotations(session_dir: Path,
                             include_unscored: bool = False) -> List[Annotation]:
    from .annotations import load_fixture_dir

    return load_fixture_dir(session_dir, include_unscored)


# ---------------------------------------------------------------------------
# baseline benchmark of the EXISTING recognizer
# ---------------------------------------------------------------------------

@dataclass
class _SplitFrames:
    role: str
    frames: list  # (FrameObservation, Annotation)
    sessions: set


def _read_frames(root: Path) -> Dict[str, _SplitFrames]:
    from PIL import Image

    from .calibration import TableCalibration
    from .pokernow import PokerNowStyleAdapter, pokernow_hu_layout

    assign = load_root_splits(root)
    out = {r: _SplitFrames(r, [], set()) for r in SPLIT_ROLES}
    for sd in session_dirs(root):
        role = assign.get(sd.name, "unassigned")
        if role not in out:
            continue
        cal_p = sd / "calibration.json"
        if cal_p.exists():
            cal = TableCalibration.from_dict(json.loads(cal_p.read_text()))
        else:
            cal = pokernow_hu_layout()
        adapter = PokerNowStyleAdapter(cal)
        for ann in load_fixture_annotations(sd):
            img = Image.open(ann.image).convert("RGB")
            obs = adapter.read_frame(img)
            out[role].frames.append((obs, ann))
            out[role].sessions.add(sd.name)
    return out


def baseline_benchmark(root: Path) -> dict:
    """Evaluate the existing recognizer per split. Splits without real data
    are reported BLOCKED — never fabricated."""
    from .annotations import score

    report: dict = {"format": DATASET_FORMAT, "splits": {}}
    frames_by_split = _read_frames(root)
    for role in SPLIT_ROLES:
        sf = frames_by_split[role]
        if not sf.frames:
            report["splits"][role] = {
                "status": "BLOCKED",
                "reason": "no annotated real sessions in this split",
            }
            continue
        s = score(sf.frames)
        report["splits"][role] = {
            "status": "ok",
            "sessions": sorted(sf.sessions),
            "frames": len(sf.frames),
            "metrics": s,
        }
    if report["splits"]["held_out"].get("status") == "BLOCKED":
        report["held_out_status"] = (
            "REAL HELD-OUT VALIDATION: BLOCKED — needs new independent "
            "labeled PokerNow sessions (docs/real_pokernow_test_checklist.md)")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="poker_alpha.observer.dataset",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("manifest", "check", "coverage", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("root", type=Path)
        p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("split")
    p.add_argument("root", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", default="0.6,0.2,0.2")
    p.add_argument("--lock", action="store_true",
                   help="lock the held-out assignment after splitting")
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        doc = build_manifest(args.root)
    elif args.cmd == "split":
        fr = tuple(float(x) for x in args.fractions.split(","))
        doc = assign_splits(args.root, seed=args.seed, fractions=fr,
                            lock=args.lock)
    elif args.cmd == "check":
        errs = check_splits(args.root)
        doc = {"clean": not errs, "problems": errs}
    elif args.cmd == "coverage":
        doc = coverage_report(args.root)
    else:
        doc = baseline_benchmark(args.root)

    text = json.dumps(doc, indent=1, default=str)
    if getattr(args, "out", None):
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.cmd == "check" and doc["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
