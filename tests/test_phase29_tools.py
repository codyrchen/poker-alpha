"""Phase 29 tooling: strategy artifacts, canonical matrix, duplicate cross-play."""

import dataclasses

import numpy as np
import pytest

from poker_alpha.games import HoldemGame
from poker_alpha.poker.cards import codes
from poker_alpha.solver_config import PRIMARY_CONFIG
from poker_alpha.solvers.strategy_artifact import (StrategyArtifactError,
                                                   export_solver, load_artifact)
from poker_alpha.validation.canonical_matrix import (CATEGORIES, FLOPS,
                                                     canonical_matrix, category,
                                                     evaluate_matrix,
                                                     representative_holes,
                                                     solver_lookup)
from poker_alpha.validation.crossplay import (calling_station, duplicate_match,
                                              table_policy, uniform_policy)


@pytest.fixture(scope="module")
def trained():
    s = PRIMARY_CONFIG.build_solver(seed=2)
    s.train(40)
    return s


def test_artifact_roundtrip_and_signature_checks(trained, tmp_path):
    path = export_solver(trained, tmp_path / "a.npz", meta={"seed": 2})
    art = load_artifact(path, PRIMARY_CONFIG.build_game())
    visited = {k for k, n in trained.infosets.items() if n.strategy_sum.sum() > 0}
    assert set(art.strategy) == visited
    k = next(iter(visited))
    probs, visits = art.lookup(k)
    node = trained.infosets[k]
    assert list(probs) == node.actions
    assert np.allclose([probs[a] for a in node.actions], node.average_strategy(), atol=1e-6)
    assert visits == pytest.approx(node.strategy_sum.sum(), rel=1e-6)
    assert art.meta["seed"] == 2 and art.meta["iterations"] == 40
    assert art.config_signature == PRIMARY_CONFIG.signature()
    with pytest.raises(StrategyArtifactError, match="solver config"):
        load_artifact(path, HoldemGame())
    other = dataclasses.replace(PRIMARY_CONFIG, raise_cap=2).build_game()
    with pytest.raises(StrategyArtifactError):
        load_artifact(path, other)
    (tmp_path / "bad.npz").write_bytes(b"junk")
    with pytest.raises(StrategyArtifactError):
        load_artifact(tmp_path / "bad.npz")


def test_matrix_deterministic_and_categories_correct():
    m1, m2 = canonical_matrix(), canonical_matrix()
    assert m1 == m2 and len(m1) > 600
    assert len({s.name for s in m1}) == len(m1)
    for bt, flop in FLOPS.items():
        for cat, hole in representative_holes(flop).items():
            assert cat in CATEGORIES
            assert category(codes(hole), codes(flop)) == cat
            assert not set(hole) & {"2c", "3d"}
    streets = {s.street for s in m1}
    assert streets == {"preflop", "flop", "turn", "river"}


def test_matrix_unvisited_reported(trained):
    game = PRIMARY_CONFIG.build_game()
    rows = evaluate_matrix(game, lambda k: None, canonical_matrix()[:20])
    assert all(r["status"] == "UNVISITED" and r["policy"] is None for r in rows)
    rows = evaluate_matrix(game, solver_lookup(trained))
    vis = [r for r in rows if r["status"] == "visited"]
    assert vis, "40 iterations should visit some preflop spots"
    for r in vis:
        assert sum(r["classes"].values()) == pytest.approx(1.0, abs=1e-3)
        assert set(r["policy"]) == set(r["legal"])


def test_duplicate_match_cancels_identical_policies():
    game = PRIMARY_CONFIG.build_game()
    r = duplicate_match(game, calling_station, calling_station, 200, seed=3)
    assert r.mean_bb_per_hand == pytest.approx(0.0, abs=1e-12)
    a = duplicate_match(game, calling_station, uniform_policy, 200, seed=3)
    b = duplicate_match(game, calling_station, uniform_policy, 200, seed=3)
    assert a == b                                  # reproducible
    assert a.a_as_button_bb_per_hand != a.a_as_bb_bb_per_hand


def test_table_policy_falls_back_to_uniform():
    game = PRIMARY_CONFIG.build_game()
    pol = table_policy(lambda k: None)
    s = game.deal(np.random.default_rng(0))
    legal = game.legal_actions(s)
    assert np.allclose(pol(game, s, legal), 1.0 / len(legal))
    assert pol.stats["misses"] == 1


def test_committed_strategy_artifact_matches_locked_config():
    from pathlib import Path

    path = Path(__file__).parents[1] / "results" / "strategy" / "holdem_v1_seed0.npz"
    art = load_artifact(path, PRIMARY_CONFIG.build_game())
    assert art.config_signature == PRIMARY_CONFIG.signature()
    assert art.meta["iterations"] >= 100000 and len(art) > 75000  # visited infosets only
    assert "IMPERFECT RECALL" in art.meta["recall"]
    for probs in list(art.strategy.values())[:2000]:
        assert abs(sum(probs.values()) - 1.0) < 1e-4
