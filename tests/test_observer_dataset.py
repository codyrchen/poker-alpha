"""Observer dataset tooling: manifests, session-level splits, leakage
checks, coverage, baseline benchmark, ML scaffold sufficiency gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from poker_alpha.observer.dataset import (assign_splits, baseline_benchmark,
                                          build_manifest, check_splits,
                                          coverage_report, load_lock,
                                          load_root_splits)
from poker_alpha.observer.ml import InsufficientData, train_cards

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "fixtures" / "pokernow"


def _annotation(image: str, hero=("As", "Kd"), pot=3.0, street="preflop"):
    return {
        "format": "pokeralpha.screenshot_annotation/v1",
        "image": image, "role": "unassigned", "status": "complete",
        "num_seats": 2, "hero_seat": 0, "dealer_seat": 1,
        "hero_cards": list(hero), "board": [], "pot": pot, "street": street,
        "seats": [{"seat": 0, "stack": 50.0, "bet": 0.0, "active": True},
                  {"seat": 1, "stack": 49.0, "bet": 1.0, "active": True}],
    }


def _make_root(tmp: Path, n_sessions: int = 5, frames_per: int = 2) -> Path:
    root = tmp / "dataset"
    cards = ["As", "Kd", "Qh", "Jc", "Ts", "9d", "8h", "7c", "6s", "5d"]
    k = 0
    for i in range(n_sessions):
        sd = root / f"session_{i:03d}"
        (sd / "annotations").mkdir(parents=True)
        (sd / "raw").mkdir()
        for f in range(frames_per):
            img = Image.new("RGB", (640, 360), (20 + k, 60, 40))
            img.save(sd / "raw" / f"frame_{f:04d}.png")
            ann = _annotation(f"raw/frame_{f:04d}.png",
                              hero=(cards[k % 10], cards[(k + 3) % 10]))
            (sd / "annotations" / f"frame_{f:04d}.json").write_text(
                json.dumps(ann))
            k += 1
    return root


def test_manifest_counts_and_hashes(tmp_path):
    root = _make_root(tmp_path)
    m = build_manifest(root)
    assert m["format"] == "pokeralpha.observer_dataset/v1"
    assert m["counts"] == {"sessions": 5, "frames": 10}
    f = m["sessions"][0]["frames"][0]
    assert len(f["annotation_sha256"]) == 64 and len(f["image_sha256"]) == 64
    assert m["problems"] == []


def test_split_is_session_level_and_deterministic(tmp_path):
    root = _make_root(tmp_path)
    a1 = assign_splits(root, seed=7)["assignments"]
    a2 = assign_splits(root, seed=7)["assignments"]
    assert a1 == a2                                   # deterministic
    assert set(a1.values()) <= {"tuning", "validation", "held_out"}
    assert "held_out" in a1.values() and "validation" in a1.values()
    # per-session splits.json gives every frame the session's role
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        per = json.loads((sd / "splits.json").read_text())
        assert set(per.values()) == {a1[sd.name]}
    assert check_splits(root) == []


def test_heldout_lock_enforced(tmp_path):
    root = _make_root(tmp_path)
    out = assign_splits(root, seed=0, lock=True)
    locked = set(out["lock"]["held_out_sessions"])
    assert locked and load_lock(root) is not None
    # re-splitting with another seed keeps the locked sessions held out
    a2 = assign_splits(root, seed=99)["assignments"]
    assert all(a2[s] == "held_out" for s in locked)
    # manual tampering is detected
    sp = json.loads((root / "splits.json").read_text())
    sp[sorted(locked)[0]] = "tuning"
    (root / "splits.json").write_text(json.dumps(sp))
    errs = check_splits(root)
    assert any("LOCK VIOLATION" in e for e in errs)


def test_check_catches_duplicate_images_across_splits(tmp_path):
    root = _make_root(tmp_path)
    assign_splits(root, seed=0)
    a = load_root_splits(root)
    tune = next(s for s, r in a.items() if r == "tuning")
    held = next(s for s, r in a.items() if r == "held_out")
    # copy a frame byte-identically from tuning into held_out
    src = root / tune / "raw" / "frame_0000.png"
    shutil.copy(src, root / held / "raw" / "copied.png")
    ann = _annotation("raw/copied.png")
    (root / held / "annotations" / "copied.json").write_text(json.dumps(ann))
    # keep per-session splits consistent so only the duplicate fires
    per = json.loads((root / held / "splits.json").read_text())
    per["copied"] = "held_out"
    (root / held / "splits.json").write_text(json.dumps(per))
    errs = check_splits(root)
    assert any("multiple splits" in e for e in errs)


def test_coverage_report(tmp_path):
    root = _make_root(tmp_path)
    c = coverage_report(root)
    assert c["frames"] == 10
    assert c["streets"] == {"preflop": 10}
    assert c["missing_coverage"]["streets_never_seen"] == ["flop", "river",
                                                           "turn"]
    assert c["card_ranks"] and c["card_suits"]
    assert "nothing about recognition accuracy" in c["note"]


@pytest.mark.vision
def test_baseline_benchmark_real_fixture_and_blocked_heldout(tmp_path):
    pytest.importorskip("scipy")
    root = tmp_path / "ds"
    root.mkdir()
    shutil.copytree(REAL, root / "pokernow_session")
    (root / "splits.json").write_text(json.dumps({"pokernow_session": "tuning"}))
    (root / "pokernow_session" / "splits.json").write_text(
        json.dumps({"hu_preflop_0001": "tuning"}))
    rep = baseline_benchmark(root)
    assert rep["splits"]["tuning"]["status"] == "ok"
    assert rep["splits"]["tuning"]["frames"] == 1
    m = rep["splits"]["tuning"]["metrics"]
    assert "hero_card" in m and "full_state" in json.dumps(m) or m
    # no real held-out data -> BLOCKED, never fabricated
    assert rep["splits"]["held_out"]["status"] == "BLOCKED"
    assert "BLOCKED" in rep["held_out_status"]


def test_ml_scaffold_refuses_insufficient_data(tmp_path):
    root = _make_root(tmp_path, n_sessions=3, frames_per=2)
    assign_splits(root, seed=0)
    with pytest.raises(InsufficientData) as e:
        train_cards(root, tmp_path / "models")
    assert "insufficient real labeled data" in str(e.value)
    assert not (tmp_path / "models").exists()         # nothing written


def test_ml_gate_is_coverage_based_not_count_based():
    """Hitting a label count must NOT pass the gate when coverage is
    missing: 400 cards that are all the same rank are untrainable."""
    from poker_alpha.observer.ml import check_sufficiency

    x = None
    crops = {"tuning": [(x, "As")] * 300, "validation": [(x, "Ah")] * 100,
             "held_out": []}
    sess = {"tuning": {"s1", "s2", "s3"}, "validation": {"s4"},
            "held_out": set()}
    problems = check_sufficiency(crops, sess)
    assert problems                                   # count alone is not enough
    text = "; ".join(problems)
    assert "ranks with <" in text                     # missing rank coverage
    assert "imbalance" in text                        # 100% aces
    assert "validation covers" in text                # no validation diversity


def test_ml_gate_passes_only_with_real_coverage():
    from poker_alpha.observer.ml import RANKS, SUITS, check_sufficiency

    x = None
    deck = [r + s for r in RANKS for s in SUITS]
    tune = [(x, deck[i % 52]) for i in range(260)]    # ~5 of every card
    val = [(x, deck[(i * 7) % 52]) for i in range(60)]
    crops = {"tuning": tune, "validation": val, "held_out": []}
    sess = {"tuning": {"s1", "s2", "s3"}, "validation": {"s4"},
            "held_out": set()}
    assert check_sufficiency(crops, sess) == []
    # losing the independent validation session fails it again
    assert check_sufficiency(crops, {**sess, "validation": set()})


def test_ml_cli_graceful_exit(tmp_path):
    root = _make_root(tmp_path, n_sessions=2, frames_per=1)
    assign_splits(root, seed=0)
    r = subprocess.run([sys.executable, "-m", "poker_alpha.observer.ml",
                        "train-cards", str(root), "--out",
                        str(tmp_path / "m")], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode != 0
    assert "insufficient real labeled data" in (r.stderr + r.stdout)


def test_dataset_cli_roundtrip(tmp_path):
    root = _make_root(tmp_path)
    for cmd in (["manifest"], ["split", "--seed", "3"], ["check"],
                ["coverage"]):
        r = subprocess.run([sys.executable, "-m",
                            "poker_alpha.observer.dataset", cmd[0],
                            str(root)] + cmd[1:],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, (cmd, r.stderr)


def test_softmax_trainer_mechanics():
    """The trainer itself learns a separable toy problem (mechanics test
    only — no accuracy claim about real card recognition)."""
    import numpy as np

    from poker_alpha.observer.ml import predict_softmax, train_softmax

    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.1, (300, 8))
    y = rng.integers(0, 3, 300)
    X[np.arange(300), y] += 2.0
    W = train_softmax(X, y, ["a", "b", "c"], epochs=200)
    assert (predict_softmax(W, X) == y).mean() > 0.95
