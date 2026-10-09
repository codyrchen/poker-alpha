"""Phase 55: strategy artifact metadata and rejection of mismatched / damaged
artifacts (no silent fallback to a mismatched strategy)."""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from poker_alpha.decision.strategy import SolverStrategyProvider
from poker_alpha.solver_config import PRIMARY_CONFIG, V2_CONFIG
from poker_alpha.solvers.strategy_artifact import (FORMAT, FORMAT_V1, StrategyArtifactError,
                                                   export_solver, load_artifact, manifest_path)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v2_artifact(tmp_path_factory):
    s = V2_CONFIG.build_solver(seed=3)
    s.train(8)
    return export_solver(s, tmp_path_factory.mktemp("a") / "v2.npz", meta={"seed": 3})


def test_v2_artifact_carries_full_provenance(v2_artifact):
    a = load_artifact(v2_artifact, V2_CONFIG.build_game())
    assert a.format == FORMAT and a.config_signature == V2_CONFIG.signature()
    c = a.config
    for k in ("config_version", "encoder", "encoder_signature", "game_signature",
              "action_abstraction", "starting_stack", "raise_cap", "averaging", "sampling",
              "bet_fractions", "preflop_raise_multiples"):
        assert k in c, k
    assert c["config_version"] == 2 and "blinds=0.5/1" in c["game_signature"]
    assert a.meta["seed"] == 3 and a.meta["iterations"] == 8
    assert a.meta["sampler"] == "MCCFRSolver"
    assert a.commit and len(a.content_sha256) == 64


@pytest.mark.parametrize("what,game", [
    ("wrong config", lambda: PRIMARY_CONFIG.build_game()),
    ("wrong action abstraction",
     lambda: dataclasses.replace(V2_CONFIG, bet_fractions=(("b50", 0.5),)).build_game()),
    ("wrong raise cap", lambda: dataclasses.replace(V2_CONFIG, raise_cap=2).build_game()),
    ("wrong stack", lambda: dataclasses.replace(V2_CONFIG, starting_stack=50.0).build_game()),
    ("incompatible encoder", lambda: dataclasses.replace(V2_CONFIG, encoder="compact").build_game()),
])
def test_mismatched_game_is_rejected(v2_artifact, what, game):
    with pytest.raises(StrategyArtifactError, match="signature mismatch"):
        load_artifact(v2_artifact, game())


def test_wrong_blinds_are_rejected_at_lookup(v2_artifact):
    """The abstract game has fixed 0.5 / 1 BB blinds (amounts are in big
    blinds); another blind ratio or an ante is refused per decision."""
    from poker_alpha.pipeline import observe_manual

    prov = SolverStrategyProvider.from_artifact(v2_artifact)
    base = {"num_seats": 2, "hero_seat": 0, "dealer": 0, "hero_cards": "As Ks", "board": "",
            "actor": 0, "pot": 1.5, "actions": [],
            "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                      {"stack": 99.0, "bet": 1.0, "committed": 1.0}]}
    for blinds in ({"small_blind": 0.4, "big_blind": 1.0},
                   {"small_blind": 0.5, "big_blind": 1.0, "ante": 0.1}):
        obs = observe_manual(dict(base, **blinds)).state
        miss = prov.lookup(obs)
        assert "blind structure differs" in getattr(miss, "reason", str(miss))


def test_corrupt_and_truncated_files_are_rejected(v2_artifact, tmp_path):
    raw = v2_artifact.read_bytes()
    (tmp_path / "trunc.npz").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(StrategyArtifactError):
        load_artifact(tmp_path / "trunc.npz")
    flipped = bytearray(raw)
    flipped[len(raw) // 3] ^= 0xFF
    (tmp_path / "flip.npz").write_bytes(bytes(flipped))
    with pytest.raises(StrategyArtifactError):
        load_artifact(tmp_path / "flip.npz")


def test_tampered_strategy_content_fails_checksum(v2_artifact, tmp_path):
    with np.load(v2_artifact, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["probs"] = d["probs"].copy()
    d["probs"][0] = 1.0 - d["probs"][0]
    np.savez_compressed(tmp_path / "t.npz", **d)
    with pytest.raises(StrategyArtifactError, match="checksum"):
        load_artifact(tmp_path / "t.npz")
    d2 = dict(d, probs=np.load(v2_artifact)["probs"],
              config_json=np.array(json.dumps(dict(json.loads(str(d["config_json"][()])),
                                                   raise_cap=9))))
    np.savez_compressed(tmp_path / "c.npz", **d2)
    with pytest.raises(StrategyArtifactError, match="does not match its signature"):
        load_artifact(tmp_path / "c.npz")


def test_committed_artifacts_have_verified_manifests(tmp_path):
    for name in ("holdem_v2_seed0", "holdem_v1_seed0"):
        p = ROOT / "results" / "strategy" / f"{name}.npz"
        m = json.loads(manifest_path(p).read_text())
        for k in ("sha256", "config_signature", "config", "encoder_signature",
                  "action_abstraction", "starting_stack", "blinds", "raise_cap", "averaging",
                  "sampler", "seed", "iterations", "generation_command",
                  "generation_commit", "first_committed_in"):
            assert k in m, (name, k)
        a = load_artifact(p)
        assert a.format == FORMAT_V1 and a.manifest["sha256"] == m["sha256"]
        assert a.config["config_version"] in (1, 2)
    rc = json.loads((ROOT / "results/validation/release_candidate.json").read_text())
    m = json.loads(manifest_path(ROOT / "results/strategy/holdem_v2_seed0.npz").read_text())
    assert m["sha256"] == rc["solver"]["artifact"]["sha256"]
    # a copy whose bytes no longer match its manifest is refused
    shutil.copy(ROOT / "results/strategy/holdem_v2_seed0.npz", tmp_path / "x.npz")
    shutil.copy(manifest_path(ROOT / "results/strategy/holdem_v2_seed0.npz"),
                manifest_path(tmp_path / "x.npz"))
    with open(tmp_path / "x.npz", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(StrategyArtifactError, match="manifest checksum"):
        load_artifact(tmp_path / "x.npz")


def test_no_silent_fallback_to_a_mismatched_strategy(v2_artifact, tmp_path):
    miss = SolverStrategyProvider.from_artifact(v2_artifact, config=PRIMARY_CONFIG)
    assert getattr(miss, "code", None) == "CONFIG_MISMATCH"
    (tmp_path / "junk.npz").write_bytes(b"not an artifact")
    miss = SolverStrategyProvider.from_artifact(tmp_path / "junk.npz")
    assert getattr(miss, "code", None) == "INCOMPATIBLE_CHECKPOINT"
