"""Phase 49 smoke test of the robustness matrix (SYNTHETIC / DERIVED ONLY)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")
pytestmark = pytest.mark.vision
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_matrix_labels_and_never_changes_confirmed_state(tmp_path, monkeypatch):
    import observer_robustness as R

    keep = {("blur", 2.0), ("scale", 0.5), ("crop_offset", 80), ("jpeg_quality", 15)}
    full = R.perturbations
    monkeypatch.setattr(R, "perturbations",
                        lambda: [p for p in full() if (p[0], p[1]) in keep])
    res = R.run(tmp_path / "r.json", n_tables=3, seed=1)
    assert "not real PokerNow accuracy" in res["label"]
    derived = [r for r in res["rows"] if r["source"] == "derived"]
    assert derived and all(r["label"].startswith("DERIVED FROM TUNING FRAME") for r in derived)
    synth = [r for r in res["rows"] if r["source"] == "synthetic"]
    assert len(synth) == 4
    assert not any(r["state_changed_during_burst"] for r in synth)
    assert json.loads((tmp_path / "r.json").read_text())["rows"]
