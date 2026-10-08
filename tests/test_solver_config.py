"""Phase 27: locked solver config, config-bound checkpoints, frozen fixtures."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.games import HoldemGame, KuhnPoker
from poker_alpha.games.holdem import HoldemState
from poker_alpha.solver_config import (PRIMARY_CONFIG, PRIMARY_SOLVER_ENCODER,
                                       HoldemSolverConfig)
from poker_alpha.solvers import CFRSolver, MCCFRSolver
from poker_alpha.solvers.holdem_analysis import spot_state
from poker_alpha.solvers.serialize import (CheckpointError, load_checkpoint,
                                           save_checkpoint)

FIX = Path(__file__).parent / "fixtures" / "solver_v1"


def test_primary_is_compact_and_flagged_imperfect_recall():
    assert PRIMARY_SOLVER_ENCODER == "compact"
    assert PRIMARY_CONFIG.encoder == "compact"
    assert PRIMARY_CONFIG.perfect_recall is False
    d = PRIMARY_CONFIG.to_dict()
    assert "recall=imperfect" in d["encoder_signature"]
    assert d["sampling"] == "external-sampling-mccfr"


def test_signature_is_stable_and_covers_every_field():
    sig = PRIMARY_CONFIG.signature()
    assert sig.startswith("HoldemSolverConfig:v1:")
    assert HoldemSolverConfig().signature() == sig
    variants = [
        dataclasses.replace(PRIMARY_CONFIG, encoder="compact_exact"),
        dataclasses.replace(PRIMARY_CONFIG, bet_fractions=(("b33", 0.33), ("b75", 0.75))),
        dataclasses.replace(PRIMARY_CONFIG, starting_stack=50.0),
        dataclasses.replace(PRIMARY_CONFIG, raise_cap=2),
        dataclasses.replace(PRIMARY_CONFIG, reference_range="other"),
    ]
    sigs = {v.signature() for v in variants}
    assert sig not in sigs and len(sigs) == len(variants)


def test_feature_table_edit_changes_signature(monkeypatch):
    from poker_alpha.abstraction import features

    sig = PRIMARY_CONFIG.signature()
    monkeypatch.setitem(features.MADE_STRENGTH, "air_high", 0)
    assert PRIMARY_CONFIG.signature() != sig


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        HoldemSolverConfig(encoder="nope")
    with pytest.raises(ValueError):
        HoldemSolverConfig(sampling="outcome-sampling")


def test_checkpoint_bound_to_config(tmp_path):
    solver = PRIMARY_CONFIG.build_solver(seed=3)
    solver.train(3)
    path = save_checkpoint(solver, tmp_path / "c.npz")
    with np.load(path, allow_pickle=False) as data:
        assert str(data["solver_config"][()]) == PRIMARY_CONFIG.signature()
    again = load_checkpoint(path, PRIMARY_CONFIG.build_game())
    assert set(again.infosets) == set(solver.infosets)
    # Same encoder and tree, but no locked config -> refused.
    bare = HoldemGame(starting_stack=100.0, bet_fractions=dict(PRIMARY_CONFIG.bet_fractions),
                      encoder=PRIMARY_CONFIG.build_game().encoder)
    with pytest.raises(CheckpointError, match="solver config"):
        load_checkpoint(path, bare)
    other = dataclasses.replace(PRIMARY_CONFIG, reference_range="other").build_game()
    with pytest.raises(CheckpointError, match="solver config"):
        load_checkpoint(path, other)


def test_format1_checkpoint_still_loads_without_config(tmp_path):
    solver = CFRSolver(KuhnPoker())
    solver.train(2)
    path = save_checkpoint(solver, tmp_path / "k.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "solver_config"}
    arrays["format_version"] = np.array(1, dtype=np.int64)
    np.savez(tmp_path / "k1.npz", **arrays)
    loaded = load_checkpoint(tmp_path / "k1.npz", KuhnPoker())
    assert set(loaded.infosets) == set(solver.infosets)
    with pytest.raises(CheckpointError, match="solver config"):
        load_checkpoint(tmp_path / "k1.npz", _with_config(KuhnPoker()))


def _with_config(game):
    game.solver_config = PRIMARY_CONFIG
    return game


def _fixture(name):
    return json.loads((FIX / name).read_text())


def test_frozen_regression_corpus_keys_unchanged():
    fx = _fixture("regression_corpus.json")
    assert fx["config_signature"] == PRIMARY_CONFIG.signature(), \
        "locked config changed: bump HoldemSolverConfig and regenerate fixtures deliberately"
    game = PRIMARY_CONFIG.build_game()
    assert fx["game_signature"] == game.signature()
    assert fx["encoder_signature"] == game.encoder_signature()
    assert len(fx["states"]) > 400
    for row in fx["states"]:
        s = HoldemState(holes=tuple(tuple(h) for h in row["holes"]),
                        board=tuple(row["board"]), streets=tuple(row["streets"]),
                        contrib=tuple(row["contrib"]))
        assert game.infoset_key(s) == row["key"], row
        assert game.legal_actions(s) == row["legal"], row


def test_frozen_corpus_generation_is_deterministic():
    from poker_alpha.validation.abstraction_audit import generate_corpus

    fx = _fixture("regression_corpus.json")
    game = PRIMARY_CONFIG.build_game()
    corpus = generate_corpus(game, fx["generator"]["hands"], fx["generator"]["seed"])
    assert [[list(h) for h in s.holes] for s in corpus] == [r["holes"] for r in fx["states"]]
    assert [list(s.streets) for s in corpus] == [r["streets"] for r in fx["states"]]


def test_canonical_suite_keys_unchanged():
    fx = _fixture("canonical_suite.json")
    assert fx["config_signature"] == PRIMARY_CONFIG.signature()
    game = PRIMARY_CONFIG.build_game()
    for row in fx["spots"]:
        s = spot_state(game, row["position"], tuple(row["hole"]), tuple(row["board"]),
                       tuple(row["streets"]))
        assert game.infoset_key(s) == row["key"], row["name"]
        assert game.legal_actions(s) == row["legal"], row["name"]
    keys = [r["key"] for r in fx["spots"]]
    assert len(set(keys)) == len(keys), "canonical spots must map to distinct infosets"


def test_selection_artifact_consistent():
    sel = json.loads((Path(__file__).parents[1] / "results" / "validation"
                      / "solver_abstraction_selection.json").read_text())
    assert sel["selected"]["PRIMARY_SOLVER_ENCODER"] == PRIMARY_SOLVER_ENCODER
    assert sel["selected"]["config_signature"] == PRIMARY_CONFIG.signature()
    assert "IMPERFECT RECALL" in sel["selected"]["recall"]
    assert sel["passing"] == ["compact"]
    assert all(sel["gate_27_to_28"].values())


def test_mccfr_unaffected_for_plain_game():
    a = MCCFRSolver(HoldemGame(), seed=1)
    a.train(2)
    assert a.game.solver_config_signature() == ""
