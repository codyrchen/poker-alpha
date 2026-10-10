"""Phase 3: exact pause/save/load/resume for CFR, CFR+ and MCCFR."""

import os
import zipfile

import numpy as np
import pytest

from poker_alpha.games import HoldemGame, KuhnPoker, LeducPoker
from poker_alpha.solvers import CFRPlusSolver, CFRSolver, MCCFRSolver
from poker_alpha.solvers.digest import strategy_digest
from poker_alpha.solvers.serialize import (CHECKPOINT_FORMAT_VERSION,
                                           CheckpointError, load_checkpoint,
                                           save_checkpoint)


def _exact(solver):
    return strategy_digest(solver.average_strategy(), quantum_bits=None)


def _assert_identical(a, b):
    assert a.iterations == b.iterations
    assert set(a.infosets) == set(b.infosets)
    for key, node in a.infosets.items():
        other = b.infosets[key]
        assert node.actions == other.actions
        np.testing.assert_array_equal(node.regret_sum, other.regret_sum)
        np.testing.assert_array_equal(node.strategy_sum, other.strategy_sum)
    assert _exact(a) == _exact(b)


@pytest.mark.parametrize("make", [
    lambda: CFRSolver(KuhnPoker()),
    lambda: CFRPlusSolver(KuhnPoker()),
    lambda: MCCFRSolver(KuhnPoker(), seed=3),
], ids=["cfr", "cfr_plus", "mccfr"])
def test_resume_is_bit_identical_kuhn(tmp_path, make):
    continuous = make()
    continuous.train(1000)

    first = make()
    first.train(500)
    path = save_checkpoint(first, tmp_path / "ckpt.npz")
    resumed = load_checkpoint(path, KuhnPoker())
    assert type(resumed) is type(first)
    assert resumed.iterations == 500
    resumed.train(500)
    _assert_identical(continuous, resumed)


def test_resume_is_bit_identical_leduc_mccfr(tmp_path):
    continuous = MCCFRSolver(LeducPoker(), seed=9)
    continuous.train(400)
    first = MCCFRSolver(LeducPoker(), seed=9)
    first.train(250)
    resumed = load_checkpoint(save_checkpoint(first, tmp_path / "l"),
                              LeducPoker())
    resumed.train(150)
    _assert_identical(continuous, resumed)


def test_negative_control_mccfr_rng_reset_changes_result(tmp_path):
    continuous = MCCFRSolver(KuhnPoker(), seed=3)
    continuous.train(1000)
    first = MCCFRSolver(KuhnPoker(), seed=3)
    first.train(500)
    resumed = load_checkpoint(save_checkpoint(first, tmp_path / "c"),
                              KuhnPoker())
    resumed.rng = np.random.default_rng(3)  # wrong: restart the stream
    resumed.train(500)
    assert _exact(resumed) != _exact(continuous)


def test_negative_control_cfr_plus_iteration_reset_changes_result(tmp_path):
    # CFR+ weights iteration t by t in the average: the counter matters.
    continuous = CFRPlusSolver(KuhnPoker())
    continuous.train(1000)
    first = CFRPlusSolver(KuhnPoker())
    first.train(500)
    resumed = load_checkpoint(save_checkpoint(first, tmp_path / "c"),
                              KuhnPoker())
    resumed.iterations = 0  # wrong: forget the iteration count
    resumed.train(500)
    assert _exact(resumed) != _exact(continuous)


def test_checkpoint_is_pickle_free_and_versioned(tmp_path):
    solver = MCCFRSolver(KuhnPoker(), seed=1)
    solver.train(50)
    path = save_checkpoint(solver, tmp_path / "x.npz")
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert "format_version.npy" in names and "rng_state.npy" in names
    with np.load(path, allow_pickle=False) as data:  # must not need pickle
        assert int(data["format_version"]) == CHECKPOINT_FORMAT_VERSION
        assert str(data["solver_type"][()]) == "MCCFR"
        assert str(data["game_signature"][()]) == "KuhnPoker:v1"
        keys = [str(k) for k in data["keys"]]
        assert keys == sorted(keys, key=lambda k: k.encode())


def test_wrong_game_rejected(tmp_path):
    solver = CFRSolver(KuhnPoker())
    solver.train(5)
    path = save_checkpoint(solver, tmp_path / "k.npz")
    with pytest.raises(CheckpointError, match="game signature"):
        load_checkpoint(path, LeducPoker())


def test_holdem_config_is_part_of_signature(tmp_path):
    solver = MCCFRSolver(HoldemGame(), seed=0)
    solver.train(2)
    path = save_checkpoint(solver, tmp_path / "h.npz")
    with pytest.raises(CheckpointError):
        load_checkpoint(path, HoldemGame(starting_stack=50))


def test_corrupt_and_wrong_version_rejected(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(bad, KuhnPoker())

    solver = CFRSolver(KuhnPoker())
    solver.train(3)
    path = save_checkpoint(solver, tmp_path / "v.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["format_version"] = np.array(999, dtype=np.int64)
    np.savez(tmp_path / "v999.npz", **arrays)
    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(tmp_path / "v999.npz", KuhnPoker())


def test_holdem_mccfr_checkpoint_smoke(tmp_path):
    continuous = MCCFRSolver(HoldemGame(), seed=4)
    continuous.train(6)
    first = MCCFRSolver(HoldemGame(), seed=4)
    first.train(3)
    path = save_checkpoint(first, tmp_path / "holdem.npz")
    assert os.path.getsize(path) > 0
    resumed = load_checkpoint(path, HoldemGame())
    resumed.train(3)
    _assert_identical(continuous, resumed)
