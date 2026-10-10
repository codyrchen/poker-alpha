"""Confidence schema v2: recent movement + strategy binding (final-trust)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from poker_alpha.decision.solver_gate import ConfidenceTable, KeyStats
from poker_alpha.solver_config import RELEASE_CONFIG


def _table(schema, meta=None):
    stats = {"0|0|AA|inr0fnoneans3|f.c.x200.x250.x350.a":
             KeyStats(500.0, 0.05, 0.1, math.nan)}
    return ConfidenceTable(RELEASE_CONFIG.signature(), stats, (),
                           meta or {}, schema=schema)


def test_v2_roundtrip(tmp_path):
    t = _table(2, {"schema": 2, "movement_mode": "recent",
                   "movement_from": 750_000, "movement_to": 1_000_000,
                   "final": 1_000_000, "seeds": [0, 1, 2]})
    p = t.save(tmp_path / "conf_v2.npz")
    loaded = ConfidenceTable.load(p, RELEASE_CONFIG.signature())
    assert loaded.schema == 2
    assert loaded.meta["movement_mode"] == "recent"
    assert len(loaded.stats) == 1


def test_v1_tables_still_load(tmp_path):
    t = _table(1, {"final": 200_000, "earlier": 10_000})
    p = t.save(tmp_path / "conf_v1.npz")
    loaded = ConfidenceTable.load(p)
    assert loaded.schema == 1
    assert loaded.meta["final"] == 200_000


def test_release_v1_table_loads_with_schema_1():
    loaded = ConfidenceTable.load(
        "results/strategy/holdem_v2_seed0_200k_confidence.npz",
        RELEASE_CONFIG.signature())
    assert loaded.schema == 1


def test_unknown_schema_rejected():
    with pytest.raises(ValueError):
        _table(3)


def test_strategy_binding_rejects_mismatch(tmp_path):
    """A v2 table bound to one artifact must not load with another."""
    from poker_alpha.decision.strategy import LookupMiss, SolverStrategyProvider
    from poker_alpha.native import NativeMCCFRSolver

    pytest.importorskip("poker_alpha_native")
    a = NativeMCCFRSolver(RELEASE_CONFIG, seed=1)
    a.train(50)
    art1 = a.export_strategy(tmp_path / "s1.npz")
    b = NativeMCCFRSolver(RELEASE_CONFIG, seed=2)
    b.train(50)
    art2 = b.export_strategy(tmp_path / "s2.npz")

    from poker_alpha.utils.provenance import file_sha256

    keys = list(a.average_strategy())[:5]
    stats = {k: KeyStats(100.0, 0.01, 0.05, math.nan) for k in keys}
    t = ConfidenceTable(RELEASE_CONFIG.signature(), stats, (),
                        {"schema": 2, "final": 50,
                         "strategy_sha256": file_sha256(art1)}, schema=2)
    conf = t.save(tmp_path / "bound.npz")

    ok = SolverStrategyProvider.from_artifact(art1, config=RELEASE_CONFIG,
                                              confidence_path=conf)
    assert not isinstance(ok, LookupMiss)
    assert ok.confidence is not None and ok.confidence.schema == 2

    bad = SolverStrategyProvider.from_artifact(art2, config=RELEASE_CONFIG,
                                               confidence_path=conf)
    assert isinstance(bad, LookupMiss)
    assert bad.code == "CONFIG_MISMATCH"
    assert "sha256" in bad.reason


def test_iteration_mismatch_rejected(tmp_path):
    from poker_alpha.decision.strategy import LookupMiss, SolverStrategyProvider
    from poker_alpha.native import NativeMCCFRSolver

    pytest.importorskip("poker_alpha_native")
    a = NativeMCCFRSolver(RELEASE_CONFIG, seed=3)
    a.train(60)
    art = a.export_strategy(tmp_path / "s.npz")
    stats = {k: KeyStats(100.0) for k in list(a.average_strategy())[:3]}
    t = ConfidenceTable(RELEASE_CONFIG.signature(), stats, (),
                        {"schema": 2, "final": 999}, schema=2)
    conf = t.save(tmp_path / "it_mismatch.npz")
    bad = SolverStrategyProvider.from_artifact(art, config=RELEASE_CONFIG,
                                               confidence_path=conf)
    assert isinstance(bad, LookupMiss) and bad.code == "CONFIG_MISMATCH"
