"""The current release strategy is the native-backend v2 seed 0 at 2M with
its SHA-bound SolverConfidence v2 table; the previous 100k and 200k releases
stay loadable (final-trust project promotion, evidence in
results/validation/final_release_comparison.json)."""

from pathlib import Path

from poker_alpha.decision.solver_gate import ConfidenceTable
from poker_alpha.pipeline import load_solver
from poker_alpha.solver_config import (PREVIOUS_RELEASE_STRATEGIES, RELEASE_CONFIDENCE,
                                       RELEASE_CONFIG, RELEASE_ITERATIONS, RELEASE_STRATEGY)
from poker_alpha.solvers.strategy_artifact import FORMAT, load_artifact
from poker_alpha.utils.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
V2_SIGNATURE = "HoldemSolverConfig:v2:733e52f1d1014e2e7973"
RELEASE_SHA256 = "f1860de75526111f9cd321e8f7f19fa94620a9856a930e5511432e5c64362da2"
CONFIDENCE_SHA256 = "ef5c32238f173d6ec3dcdd8c68c8cae6f406ea863bcee3797e2102764945b422"


def test_release_config_is_unchanged_v2():
    assert RELEASE_CONFIG.signature() == V2_SIGNATURE


def test_release_artifact_is_native_seed0_2m_and_valid():
    path = ROOT / RELEASE_STRATEGY
    assert file_sha256(path) == RELEASE_SHA256
    art = load_artifact(path, RELEASE_CONFIG.build_game())   # verifies content SHA-256
    assert art.format == FORMAT
    assert art.meta["iterations"] == RELEASE_ITERATIONS == 2_000_000
    assert art.meta["seed"] == 0
    assert art.meta["backend"] == "native"
    assert art.config_signature == RELEASE_CONFIG.signature() == V2_SIGNATURE


def test_release_confidence_table_is_bound_schema2():
    path = ROOT / RELEASE_CONFIDENCE
    assert path.name == Path(RELEASE_STRATEGY).stem + "_confidence.npz"
    assert file_sha256(path) == CONFIDENCE_SHA256
    table = ConfidenceTable.load(path, RELEASE_CONFIG.signature())
    assert table.schema == 2
    assert table.meta["final"] == RELEASE_ITERATIONS
    assert table.meta["seeds"] == [0, 1, 2]
    assert table.meta["movement_mode"] == "recent"
    assert table.meta["movement_from"] == 1_500_000
    # The table binds to the release artifact's SHA (content identical to the
    # preserved candidate copy it was built against).
    assert table.meta["strategy_sha256"] == RELEASE_SHA256


def test_loaded_release_solver_uses_the_2m_table():
    prov = load_solver(ROOT / RELEASE_STRATEGY)
    assert not hasattr(prov, "code"), prov
    assert prov.confidence is not None
    assert prov.confidence.schema == 2
    assert prov.confidence.meta["final"] == RELEASE_ITERATIONS
    assert prov.confidence.config_signature == RELEASE_CONFIG.signature()
    assert "2000000 iterations" in prov.description


def test_release_rejects_mismatched_confidence():
    """The bound table must not load with a different artifact."""
    from poker_alpha.decision.strategy import LookupMiss, SolverStrategyProvider

    other = ROOT / "results/strategy/holdem_v2_seed0_200k.npz"
    bad = SolverStrategyProvider.from_artifact(
        other, config=RELEASE_CONFIG,
        confidence_path=ROOT / RELEASE_CONFIDENCE)
    assert isinstance(bad, LookupMiss)
    assert bad.code == "CONFIG_MISMATCH"


def test_app_and_demo_load_the_release_strategy():
    from poker_alpha import platform_demo

    assert platform_demo.DEFAULT_STRATEGY == ROOT / RELEASE_STRATEGY
    src = (ROOT / "poker_alpha/ui/app.py").read_text()
    assert "STRATEGY = ROOT / RELEASE_STRATEGY" in src


def test_previous_releases_are_preserved_and_loadable():
    game = RELEASE_CONFIG.build_game()
    art_f, conf_f = PREVIOUS_RELEASE_STRATEGIES["v2@100k"]
    art = load_artifact(ROOT / art_f, game)
    assert art.meta["iterations"] == 100_000
    assert ConfidenceTable.load(ROOT / conf_f).meta["final"] == 100_000
    art_f, conf_f = PREVIOUS_RELEASE_STRATEGIES["v2@200k"]
    art = load_artifact(ROOT / art_f, game)
    assert art.meta["iterations"] == 200_000
    table = ConfidenceTable.load(ROOT / conf_f)
    assert table.schema == 1 and table.meta["final"] == 200_000
