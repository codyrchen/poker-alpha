"""The current release strategy is v2 seed 0 at 200k with its matching
200k confidence table; the previous 100k release stays loadable."""

import json
from pathlib import Path

from poker_alpha.decision.solver_gate import ConfidenceTable
from poker_alpha.pipeline import load_solver
from poker_alpha.solver_config import (PREVIOUS_RELEASE_STRATEGIES, RELEASE_CONFIDENCE,
                                       RELEASE_CONFIG, RELEASE_ITERATIONS, RELEASE_STRATEGY)
from poker_alpha.solvers.strategy_artifact import FORMAT, load_artifact
from poker_alpha.utils.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
V2_SIGNATURE = "HoldemSolverConfig:v2:733e52f1d1014e2e7973"
CANDIDATE_MANIFEST = ROOT / "results/strategy/candidates/v2_extension/MANIFEST.json"


def test_release_config_is_unchanged_v2():
    assert RELEASE_CONFIG.signature() == V2_SIGNATURE


def test_release_artifact_is_seed0_200k_and_valid():
    path = ROOT / RELEASE_STRATEGY
    art = load_artifact(path, RELEASE_CONFIG.build_game())   # verifies content SHA-256
    assert art.format == FORMAT
    assert art.meta["iterations"] == RELEASE_ITERATIONS == 200_000
    assert art.meta["seed"] == 0
    assert art.config_signature == RELEASE_CONFIG.signature() == V2_SIGNATURE
    committed = {a["file"]: a for a in json.loads(CANDIDATE_MANIFEST.read_text())
                 ["committed_artifacts"]}
    assert file_sha256(path) == committed[path.name]["sha256"]


def test_release_confidence_table_is_the_matching_200k_table():
    path = ROOT / RELEASE_CONFIDENCE
    assert path.name == Path(RELEASE_STRATEGY).stem + "_confidence.npz"
    table = ConfidenceTable.load(path, RELEASE_CONFIG.signature())
    assert table.config_signature == V2_SIGNATURE
    assert table.meta["final"] == RELEASE_ITERATIONS
    assert table.meta["seeds"] == [0, 1, 2]
    committed = {a["file"]: a for a in json.loads(CANDIDATE_MANIFEST.read_text())
                 ["committed_artifacts"]}
    assert file_sha256(path) == committed[path.name]["sha256"]


def test_loaded_release_solver_uses_the_200k_table():
    prov = load_solver(ROOT / RELEASE_STRATEGY)
    assert not hasattr(prov, "code"), prov
    assert prov.confidence is not None
    assert prov.confidence.meta["final"] == RELEASE_ITERATIONS
    assert prov.confidence.config_signature == RELEASE_CONFIG.signature()
    assert "200000 iterations" in prov.description


def test_app_and_demo_load_the_release_strategy():
    from poker_alpha import platform_demo

    assert platform_demo.DEFAULT_STRATEGY == ROOT / RELEASE_STRATEGY
    src = (ROOT / "poker_alpha/ui/app.py").read_text()
    assert "STRATEGY = ROOT / RELEASE_STRATEGY" in src


def test_previous_100k_release_is_preserved_and_loadable():
    art_f, conf_f = PREVIOUS_RELEASE_STRATEGIES["v2@100k"]
    art = load_artifact(ROOT / art_f, RELEASE_CONFIG.build_game())
    assert art.meta["iterations"] == 100_000
    assert ConfidenceTable.load(ROOT / conf_f).meta["final"] == 100_000
