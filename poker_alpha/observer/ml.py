"""Optional ML training scaffold for weak observer components.

    python -m poker_alpha.observer.ml train-cards <dataset_root> [--out DIR]

This does NOT replace the template recognizer. It exists so that once
enough REAL labeled data has been collected (see
:mod:`poker_alpha.observer.dataset`), a small card rank/suit classifier can
be trained and compared against the current recognizer baseline — with
versioned provenance, deterministic session-level splits and the held-out
set untouched until a final evaluation.

Data sufficiency is checked FIRST and the command fails gracefully with
"insufficient real labeled data" when the dataset is too small. It never
trains on synthetic fixtures and claims success.

Model artifacts are a plain ``.npz`` (softmax regression on normalized
card-crop pixels — deliberately small and inspectable) plus a
``pokeralpha.model/v1`` metadata JSON with: model type, dataset/split hash,
training config, git commit, timestamp, validation metrics, held-out status
and the artifact SHA. No opaque weights without provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_FORMAT = "pokeralpha.model/v1"

# Data sufficiency thresholds (documented; tests pin them). Below these the
# scaffold refuses to train: a classifier fitted on a handful of frames from
# one session would memorize that session's rendering, not learn cards.
MIN_LABELED_CARDS = 200
MIN_SESSIONS = 4

CROP = (24, 18)          # (h, w) normalized card crop fed to the classifier
RANKS = "AKQJT98765432"
SUITS = "shdc"


class InsufficientData(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"insufficient real labeled data: {message}")


# ---------------------------------------------------------------------------
# dataset -> labeled crops
# ---------------------------------------------------------------------------

def gather_card_crops(root: Path) -> Tuple[Dict[str, list], Dict[str, set]]:
    """Labeled hero-card crops per split: {role: [(x, label)]}, plus the
    sessions contributing to each split. Uses each session's calibration to
    locate the hero card regions."""
    from PIL import Image

    from .calibration import TableCalibration
    from .dataset import load_root_splits, session_dirs
    from .dataset import load_fixture_annotations
    from .pokernow import pokernow_hu_layout

    assign = load_root_splits(root)
    out: Dict[str, list] = {"tuning": [], "validation": [], "held_out": []}
    sess: Dict[str, set] = {k: set() for k in out}
    for sd in session_dirs(root):
        role = assign.get(sd.name)
        if role not in out:
            continue
        cal_p = sd / "calibration.json"
        cal = TableCalibration.from_dict(json.loads(cal_p.read_text())) \
            if cal_p.exists() else pokernow_hu_layout()
        regions = dict(cal.regions)
        for ann in load_fixture_annotations(sd):
            if not ann.hero_cards:
                continue
            with Image.open(ann.image).convert("L") as im:
                for i, label in enumerate(ann.hero_cards):
                    reg = regions.get(f"hero_card_{i}")
                    if reg is None or label is None:
                        continue
                    box = (int(reg.x * im.width), int(reg.y * im.height),
                           int((reg.x + reg.w) * im.width),
                           int((reg.y + reg.h) * im.height))
                    crop = im.crop(box).resize((CROP[1], CROP[0]))
                    x = np.asarray(crop, dtype=np.float32) / 255.0
                    out[role].append((x.ravel(), label))
                    sess[role].add(sd.name)
    return out, sess


# ---------------------------------------------------------------------------
# a deliberately small softmax classifier (numpy only)
# ---------------------------------------------------------------------------

def train_softmax(X: np.ndarray, y: np.ndarray, classes: List[str],
                  epochs: int = 300, lr: float = 0.5, seed: int = 0,
                  l2: float = 1e-4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k, d = len(classes), X.shape[1]
    W = rng.normal(0, 0.01, (d + 1, k)).astype(np.float64)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    Y = np.zeros((X.shape[0], k))
    Y[np.arange(len(y)), y] = 1.0
    for _ in range(epochs):
        z = Xb @ W
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = Xb.T @ (p - Y) / len(y) + l2 * W
        W -= lr * g
    return W


def predict_softmax(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.argmax(Xb @ W, axis=1)


# ---------------------------------------------------------------------------
# training command with provenance
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    from .session import git_commit

    return git_commit()


def _split_hash(root: Path) -> str:
    from .dataset import load_root_splits

    return hashlib.sha256(
        json.dumps(load_root_splits(root), sort_keys=True).encode()
    ).hexdigest()[:16]


def train_cards(root: Path, out_dir: Path, seed: int = 0,
                epochs: int = 300) -> dict:
    crops, sess = gather_card_crops(root)
    n_train = len(crops["tuning"])
    n_val = len(crops["validation"])
    n_sessions = len(sess["tuning"] | sess["validation"])
    if n_train + n_val < MIN_LABELED_CARDS or n_sessions < MIN_SESSIONS:
        raise InsufficientData(
            f"{n_train + n_val} labeled cards across {n_sessions} "
            f"train/validation sessions; need >= {MIN_LABELED_CARDS} cards "
            f"from >= {MIN_SESSIONS} sessions. Collect and annotate real "
            f"sessions first (python -m poker_alpha.observer.dataset).")
    classes = sorted({lbl for lst in (crops["tuning"], crops["validation"])
                      for _, lbl in lst})
    idx = {c: i for i, c in enumerate(classes)}
    Xt = np.stack([x for x, _ in crops["tuning"]])
    yt = np.array([idx[l] for _, l in crops["tuning"]])
    Xv = np.stack([x for x, _ in crops["validation"]])
    yv = np.array([idx[l] for _, l in crops["validation"]])
    W = train_softmax(Xt, yt, classes, epochs=epochs, seed=seed)
    val_acc = float((predict_softmax(W, Xv) == yv).mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "card_classifier.npz"
    np.savez_compressed(artifact, W=W, classes=np.array(classes),
                        crop=np.array(CROP))
    meta = {
        "format": MODEL_FORMAT,
        "model_type": "softmax-regression card classifier (numpy)",
        "task": "hero card rank+suit classification from calibrated crops",
        "dataset_root": str(root),
        "split_hash": _split_hash(root),
        "train_cards": n_train, "validation_cards": n_val,
        "sessions": sorted(sess["tuning"] | sess["validation"]),
        "config": {"crop": CROP, "epochs": epochs, "lr": 0.5, "l2": 1e-4,
                   "seed": seed},
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_accuracy": val_acc,
        "held_out_status": ("UNTOUCHED — evaluate once, at the end, via "
                            "poker_alpha.observer.dataset baseline"),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "claims": ("validation accuracy on the recorded sessions above; no "
                   "claim about other tables, themes or resolutions"),
    }
    (out_dir / "card_classifier.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="poker_alpha.observer.ml",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-cards")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/models"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args(argv)
    meta = train_cards(args.root, args.out, seed=args.seed, epochs=args.epochs)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
