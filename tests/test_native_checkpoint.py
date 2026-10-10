"""Native checkpoint, determinism, RNG and artifact tests
(Phases 9, 18, 27-30, 76, 87-89).
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_alpha.solver_config import RELEASE_CONFIG, PRIMARY_CONFIG

from native_helpers import native, native_cfg  # noqa: F401 (skip guard)

from poker_alpha.native import NativeMCCFRSolver
from poker_alpha.native.backend import NativeBackendError


def _state_tuple(solver):
    keys_u64, keys, offsets, tokens, regret, strategy, rng = \
        solver._state_arrays()
    return (keys, offsets.tolist(), tokens.tolist(), regret.tolist(),
            strategy.tolist(), rng.tolist(), solver.iterations)


# -- determinism (Phase 18) --------------------------------------------------

def test_same_seed_is_deterministic():
    a = NativeMCCFRSolver(RELEASE_CONFIG, seed=5)
    b = NativeMCCFRSolver(RELEASE_CONFIG, seed=5)
    a.train(200)
    b.train(200)
    assert _state_tuple(a) == _state_tuple(b)


def test_different_seeds_differ():
    a = NativeMCCFRSolver(RELEASE_CONFIG, seed=1)
    b = NativeMCCFRSolver(RELEASE_CONFIG, seed=2)
    a.train(50)
    b.train(50)
    assert _state_tuple(a) != _state_tuple(b)


# -- dealing statistics (Phase 9) -------------------------------------------

def test_deal_statistics():
    """Uniform card frequency, no duplicates, across 40k root deals."""
    counts = np.asarray(native.debug_deal_counts(native_cfg(RELEASE_CONFIG),
                                                 40_000, 7))
    assert counts.shape == (4, 52)
    assert counts.sum() == 160_000
    per_card = counts.sum(axis=0)
    expect = 160_000 / 52
    # Binomial std ~ sqrt(n*p*(1-p)) ~ 54.8; allow 5 sigma.
    assert np.all(np.abs(per_card - expect) < 5 * np.sqrt(expect)), per_card
    per_suit = per_card.reshape(4, 13).sum(axis=1)
    assert np.all(np.abs(per_suit - 40_000) < 5 * np.sqrt(40_000 * 0.25))
    per_rank = per_card.reshape(4, 13).sum(axis=0)
    assert np.all(np.abs(per_rank - 160_000 / 13) < 5 * np.sqrt(160_000 / 13))


# -- checkpoint resume (Phases 18, 29) ---------------------------------------

def test_checkpoint_exact_resume(tmp_path):
    full = NativeMCCFRSolver(RELEASE_CONFIG, seed=3)
    full.train(300)

    part = NativeMCCFRSolver(RELEASE_CONFIG, seed=3)
    part.train(180)
    path = part.save_checkpoint(tmp_path / "ckpt.npz")
    resumed = NativeMCCFRSolver.load_checkpoint(path, RELEASE_CONFIG)
    assert resumed.iterations == 180
    resumed.train(120)
    assert _state_tuple(resumed) == _state_tuple(full)


def test_checkpoint_roundtrip_is_identity(tmp_path):
    s = NativeMCCFRSolver(RELEASE_CONFIG, seed=4)
    s.train(100)
    path = s.save_checkpoint(tmp_path / "rt.npz")
    loaded = NativeMCCFRSolver.load_checkpoint(path, RELEASE_CONFIG)
    assert _state_tuple(loaded) == _state_tuple(s)


# -- checkpoint corruption / mismatch (Phase 87) -----------------------------

@pytest.fixture()
def saved_checkpoint(tmp_path):
    s = NativeMCCFRSolver(RELEASE_CONFIG, seed=6)
    s.train(40)
    return s.save_checkpoint(tmp_path / "c.npz")


def test_wrong_config_rejected(saved_checkpoint):
    with pytest.raises(NativeBackendError, match="config mismatch"):
        NativeMCCFRSolver.load_checkpoint(saved_checkpoint, PRIMARY_CONFIG)


def test_truncated_rejected(saved_checkpoint, tmp_path):
    data = saved_checkpoint.read_bytes()
    bad = tmp_path / "trunc.npz"
    bad.write_bytes(data[: len(data) // 2])
    with pytest.raises(NativeBackendError):
        NativeMCCFRSolver.load_checkpoint(bad, RELEASE_CONFIG)


def test_bitflip_rejected(saved_checkpoint, tmp_path):
    data = bytearray(saved_checkpoint.read_bytes())
    # Flip a byte deep in the arrays region (headers are near the front).
    data[len(data) // 2] ^= 0xFF
    bad = tmp_path / "flip.npz"
    bad.write_bytes(bytes(data))
    with pytest.raises(NativeBackendError):
        NativeMCCFRSolver.load_checkpoint(bad, RELEASE_CONFIG)


def test_not_a_checkpoint_rejected(tmp_path):
    p = tmp_path / "junk.npz"
    np.savez(p, format=np.array("something/else"))
    with pytest.raises(NativeBackendError):
        NativeMCCFRSolver.load_checkpoint(p, RELEASE_CONFIG)


# -- strategy artifact roundtrip (Phases 27, 89, 90) --------------------------

def test_artifact_export_load_decision_lookup(tmp_path):
    from poker_alpha.solvers.strategy_artifact import load_artifact

    s = NativeMCCFRSolver(RELEASE_CONFIG, seed=8)
    s.train(400)
    path = s.export_strategy(tmp_path / "native_strategy.npz",
                             meta={"note": "test export"})
    game = RELEASE_CONFIG.build_game()
    art = load_artifact(path, game)        # full checksum + signature checks
    assert art.config_signature == RELEASE_CONFIG.signature()
    assert art.meta["backend"] == "native"
    assert art.meta["iterations"] == 400
    assert art.meta["sampler"] == "NativeMCCFRSolver"
    assert len(art) > 0

    # Every exported distribution matches the solver's average strategy
    # (float32 artifact precision).
    avg = s.average_strategy()
    for key, probs in art.strategy.items():
        for action, p in probs.items():
            assert abs(p - avg[key][action]) < 1e-6
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    # The DecisionEngine-side provider loads it without native involvement.
    from poker_alpha.decision.strategy import SolverStrategyProvider

    provider = SolverStrategyProvider.from_artifact(path, config=RELEASE_CONFIG)
    assert not isinstance(provider, tuple)
    assert hasattr(provider, "lookup")


def test_artifact_matches_python_artifact_format(tmp_path):
    """Python- and native-trained artifacts are structurally identical."""
    from poker_alpha.solvers.strategy_artifact import export_solver, load_artifact

    py = RELEASE_CONFIG.build_solver(seed=9)
    for _ in range(60):
        py.iterate()
    nat = NativeMCCFRSolver(RELEASE_CONFIG, seed=9)
    nat.train(60)
    p1 = export_solver(py, tmp_path / "py.npz")
    p2 = nat.export_strategy(tmp_path / "nat.npz")
    a1 = load_artifact(p1, RELEASE_CONFIG.build_game())
    a2 = load_artifact(p2, RELEASE_CONFIG.build_game())
    assert a1.format == a2.format
    assert a1.config_signature == a2.config_signature
    # Different RNGs -> different trajectories; but the key universe overlaps
    # heavily and every shared key has the same action menu.
    shared = set(a1.strategy) & set(a2.strategy)
    assert len(shared) > 100
    for key in shared:
        assert set(a1.strategy[key]) == set(a2.strategy[key]), key
