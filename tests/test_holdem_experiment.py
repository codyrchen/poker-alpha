"""Phase 7: heads-up Hold'em MCCFR experiment tooling and proxies."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.games import HoldemGame
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.holdem_analysis import (cross_play,
                                                 describe_strategy_at,
                                                 infoset_visits,
                                                 solver_metrics, spot_state,
                                                 strategy_l1_change,
                                                 top_infoset_stability)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained():
    solver = MCCFRSolver(HoldemGame(), seed=0)
    solver.train(30)
    return solver


def test_metrics_are_consistent(trained):
    m = solver_metrics(trained)
    assert m.iterations == 30 and m.infosets == len(trained.infosets)
    assert m.memory_bytes_estimate > 16 * m.infosets
    assert 0.0 <= m.mean_entropy_bits <= np.log2(6) + 1e-9
    visits = infoset_visits(trained)
    assert m.visits_total == pytest.approx(sum(visits.values()))
    assert 0.0 <= m.singleton_fraction <= 1.0


def test_l1_change_properties():
    a = {"k": {"x": 1.0, "y": 0.0}, "j": {"x": 0.5, "y": 0.5}}
    b = {"k": {"x": 0.0, "y": 1.0}, "j": {"x": 0.5, "y": 0.5}, "new": {"x": 1.0}}
    assert strategy_l1_change(a, a) == 0.0
    assert strategy_l1_change(a, b) == pytest.approx(1.0)   # (2 + 0) / 2
    assert strategy_l1_change(a, b, {"k": 3.0, "j": 1.0}) == pytest.approx(1.5)
    assert top_infoset_stability(a, b, {"k": 5, "j": 1}, k=1) == pytest.approx(2.0)
    assert np.isnan(strategy_l1_change({}, b))


def test_cross_play_is_seeded_and_symmetric(trained):
    game = HoldemGame()
    s = trained.average_strategy()
    r1 = cross_play(game, s, {}, hands=40, seed=1)
    r2 = cross_play(game, s, {}, hands=40, seed=1)
    assert r1 == r2
    assert r1.hands == 40 and r1.std_error > 0
    # Uniform vs uniform with identical deals both seats: zero-sum mirror.
    r0 = cross_play(game, {}, {}, hands=30, seed=2)
    assert abs(r0.mean_bb_per_hand) < 5 * r0.std_error + 1e-9


def test_describe_strategy_at_human_readable_spot(trained):
    game = HoldemGame()
    rep = describe_strategy_at(game, trained.average_strategy(),
                               infoset_visits(trained), "BTN", ("As", "Ks"))
    labels = [label for _, label, _ in rep.actions]
    assert labels == ["fold", "call", "raise50", "raise100", "raise200", "all-in"]
    assert sum(p for *_, p in rep.actions) == pytest.approx(1.0)
    assert rep.infoset_key == "0|50,51||"
    assert "fold" in rep.format()
    # Big blind facing a limp: check rather than call, no fold.
    bb = describe_strategy_at(game, {}, {}, "BB", ("Qh", "Qd"),
                              streets=("c",))
    assert [label for _, label, _ in bb.actions][:2] == ["check", "bet50"]
    with pytest.raises(ValueError):
        spot_state(game, "BB", ("As", "Ks"))  # BTN acts first preflop


def test_experiment_cli_checkpoints_and_resumes(tmp_path):
    ckpt = tmp_path / "ck.npz"
    cmd = [sys.executable, str(ROOT / "experiments" / "holdem_mccfr.py"),
           "--encoder", "raw", "--stack-bb", "50", "--seed", "3",
           "--checkpoint", str(ckpt), "--checkpoint-every", "3",
           "--crossplay-hands", "4", "--outdir", str(tmp_path)]
    out = subprocess.run(cmd + ["--iterations", "6"], capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout
    assert ckpt.exists() and "iteration=6" in out
    assert (tmp_path / "data" / "holdem_mccfr_raw.csv").exists()
    out2 = subprocess.run(cmd + ["--iterations", "9"], capture_output=True,
                          text=True, cwd=ROOT, check=True).stdout
    assert "resumed" in out2 and "iteration=9" in out2
    assert "BTN 50 BB AsKs preflop unopened" in out2
