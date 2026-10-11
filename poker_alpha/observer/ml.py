"""Optional ML training scaffold for weak observer components.

    python -m poker_alpha.observer.ml train-cards <dataset_root> [--out DIR]

This does NOT replace the template recognizer. It exists so that once
enough REAL labeled data has been collected (see
:mod:`poker_alpha.observer.dataset`), a card classifier can be trained and
compared against the current recognizer baseline — with versioned
provenance, deterministic session-level splits and the held-out set
untouched until a final evaluation.

Architecture: TWO heads — a 13-way rank classifier and a 4-way suit
classifier — rather than one 52-class model. For small calibrated crops
this is the defensible choice: suit is mostly a color/shape cue and rank a
glyph, the factorized heads need coverage of 13 + 4 classes instead of 52,
and per-class support stays meaningful at realistic dataset sizes.

Data sufficiency is a COVERAGE check, not a count threshold. Hitting a
number of labels does not make the training problem identifiable; the gate
requires (documented in ``SufficiencyRules``, pinned by tests):

* >= 3 tuning sessions and >= 1 independent validation session
  (session-level splits; one session's rendering must not be both
  teacher and examiner);
* every rank seen >= 5 times and every suit >= 10 times in tuning
  (otherwise some classes are untrainable);
* validation covers >= 8 distinct ranks and >= 3 suits (otherwise it
  cannot detect per-class failure);
* no severe imbalance (most frequent rank <= 50% of tuning labels);
* and a floor of 200 labeled cards overall — a floor, not a guarantee.

These are collection targets that make the problem well-posed, NOT a
validated sample-complexity bound; accuracy claims only ever come from
the dataset baseline command on real held-out sessions.

If the gate fails the command exits with "insufficient real labeled
data: …" listing every unmet requirement. It never trains on synthetic
fixtures and claims success.

Model artifacts are plain ``.npz`` weights plus a ``pokeralpha.model/v1``
metadata JSON: model type, dataset/split hash, config, git commit,
timestamp, per-head validation metrics with per-class support, held-out
status and the artifact SHA. No opaque weights without provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_FORMAT = "pokeralpha.model/v1"

CROP = (24, 18)          # (h, w) normalized card crop fed to the classifier
RANKS = "AKQJT98765432"
SUITS = "shdc"


@dataclass(frozen=True)
class SufficiencyRules:
    """The documented coverage gate (see module docstring)."""

    min_tuning_sessions: int = 3
    min_validation_sessions: int = 1
    min_per_rank: int = 5            # tuning examples per rank
    min_per_suit: int = 10           # tuning examples per suit
    min_val_ranks: int = 8           # distinct ranks validation must cover
    min_val_suits: int = 3
    max_rank_share: float = 0.5      # imbalance guard on tuning ranks
    min_total_cards: int = 200       # overall floor (a floor, not sufficiency)


RULES = SufficiencyRules()


class InsufficientData(SystemExit):
    def __init__(self, problems: List[str]):
        super().__init__("insufficient real labeled data: "
                         + "; ".join(problems))
        self.problems = problems


def check_sufficiency(crops: Dict[str, list], sess: Dict[str, set],
                      rules: SufficiencyRules = RULES) -> List[str]:
    """Every unmet requirement, as human text. Empty list = trainable."""
    problems: List[str] = []
    tune, val = crops["tuning"], crops["validation"]
    total = len(tune) + len(val)
    if total < rules.min_total_cards:
        problems.append(f"{total} labeled cards overall; floor is "
                        f"{rules.min_total_cards}")
    if len(sess["tuning"]) < rules.min_tuning_sessions:
        problems.append(f"{len(sess['tuning'])} tuning sessions; need >= "
                        f"{rules.min_tuning_sessions}")
    if len(sess["validation"]) < rules.min_validation_sessions:
        problems.append(f"{len(sess['validation'])} validation sessions; "
                        f"need >= {rules.min_validation_sessions} "
                        f"(independent of tuning)")
    t_ranks = Counter(l[0] for _, l in tune)
    t_suits = Counter(l[1] for _, l in tune)
    missing_r = [r for r in RANKS if t_ranks.get(r, 0) < rules.min_per_rank]
    if missing_r:
        problems.append(f"ranks with < {rules.min_per_rank} tuning examples: "
                        f"{''.join(missing_r)}")
    missing_s = [s for s in SUITS if t_suits.get(s, 0) < rules.min_per_suit]
    if missing_s:
        problems.append(f"suits with < {rules.min_per_suit} tuning examples: "
                        f"{''.join(missing_s)}")
    if tune and max(t_ranks.values()) > rules.max_rank_share * len(tune):
        top = t_ranks.most_common(1)[0]
        problems.append(f"severe rank imbalance: {top[0]} is "
                        f"{top[1] / len(tune):.0%} of tuning labels")
    v_ranks = {l[0] for _, l in val}
    v_suits = {l[1] for _, l in val}
    if len(v_ranks) < rules.min_val_ranks:
        problems.append(f"validation covers {len(v_ranks)} ranks; need >= "
                        f"{rules.min_val_ranks} to detect per-class failure")
    if len(v_suits) < rules.min_val_suits:
        problems.append(f"validation covers {len(v_suits)} suits; need >= "
                        f"{rules.min_val_suits}")
    return problems


# ---------------------------------------------------------------------------
# dataset -> labeled crops
# ---------------------------------------------------------------------------

def gather_card_crops(root: Path) -> Tuple[Dict[str, list], Dict[str, set]]:
    """Labeled hero-card crops per split: {role: [(x, label)]}, plus the
    sessions contributing to each split. Uses each session's calibration to
    locate the hero card regions."""
    from PIL import Image

    from .calibration import TableCalibration
    from .dataset import (load_fixture_annotations, load_root_splits,
                          session_dirs)
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


def _head_metrics(W: np.ndarray, X: np.ndarray, y: np.ndarray,
                  classes: List[str]) -> dict:
    pred = predict_softmax(W, X)
    per_class = {}
    for i, c in enumerate(classes):
        mask = y == i
        per_class[c] = {"support": int(mask.sum()),
                        "accuracy": float((pred[mask] == i).mean())
                        if mask.any() else None}
    return {"accuracy": float((pred == y).mean()), "per_class": per_class}


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
    problems = check_sufficiency(crops, sess)
    if problems:
        raise InsufficientData(problems)

    def head(selector, classes):
        idx = {c: i for i, c in enumerate(classes)}
        Xt = np.stack([x for x, _ in crops["tuning"]])
        yt = np.array([idx[selector(l)] for _, l in crops["tuning"]])
        Xv = np.stack([x for x, _ in crops["validation"]])
        yv = np.array([idx[selector(l)] for _, l in crops["validation"]])
        W = train_softmax(Xt, yt, classes, epochs=epochs, seed=seed)
        return W, _head_metrics(W, Xv, yv, classes)

    W_rank, rank_metrics = head(lambda l: l[0], list(RANKS))
    W_suit, suit_metrics = head(lambda l: l[1], list(SUITS))

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "card_classifier.npz"
    np.savez_compressed(artifact, W_rank=W_rank, W_suit=W_suit,
                        ranks=np.array(list(RANKS)),
                        suits=np.array(list(SUITS)), crop=np.array(CROP))
    meta = {
        "format": MODEL_FORMAT,
        "model_type": ("factorized softmax card classifier (numpy): "
                       "13-way rank head + 4-way suit head"),
        "task": "hero card recognition from calibrated crops",
        "dataset_root": str(root),
        "split_hash": _split_hash(root),
        "train_cards": len(crops["tuning"]),
        "validation_cards": len(crops["validation"]),
        "sessions": sorted(sess["tuning"] | sess["validation"]),
        "sufficiency_rules": RULES.__dict__,
        "config": {"crop": CROP, "epochs": epochs, "lr": 0.5, "l2": 1e-4,
                   "seed": seed},
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation": {"rank": rank_metrics, "suit": suit_metrics},
        "held_out_status": ("UNTOUCHED — evaluate once, at the end, via "
                            "poker_alpha.observer.dataset baseline"),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "claims": ("validation metrics on the recorded sessions above; no "
                   "claim about other tables, themes, resolutions, or the "
                   "held-out split"),
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
