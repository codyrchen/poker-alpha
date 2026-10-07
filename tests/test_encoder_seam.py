"""Phase 4: information-state encoder seam for Hold'em."""

import numpy as np
import pytest

from poker_alpha.abstraction import (InformationStateEncoder,
                                     RawHoldemEncoder, ToyHoldemEncoder)
from poker_alpha.games import HoldemGame
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.serialize import (CheckpointError, load_checkpoint,
                                           save_checkpoint)


def legacy_key(game, state):
    """The pre-seam HoldemGame.infoset_key, copied verbatim."""
    player = game.current_player(state)
    hole = ",".join(str(c) for c in sorted(state.holes[player]))
    board = ",".join(str(c) for c in state.board)
    history = "/".join(state.streets)
    return f"{player}|{hole}|{board}|{history}"


def _random_decision_states(game, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = game.deal(rng)
        while not game.is_terminal(s):
            if game.is_chance(s):
                s = game.sample_chance(s, rng)
                continue
            out.append(s)
            acts = game.legal_actions(s)
            s = game.next_state(s, acts[int(rng.integers(len(acts)))])
    return out


def test_encoders_satisfy_protocol():
    assert isinstance(RawHoldemEncoder(), InformationStateEncoder)
    assert isinstance(ToyHoldemEncoder(), InformationStateEncoder)


def test_raw_encoder_output_unchanged():
    game = HoldemGame()
    for s in _random_decision_states(game, 400, seed=1):
        assert game.infoset_key(s) == legacy_key(game, s)
        assert RawHoldemEncoder().encode(game, s) == legacy_key(game, s)


def test_signatures_distinct_and_deterministic():
    assert RawHoldemEncoder().signature() == "RawHoldemEncoder:v1"
    assert ToyHoldemEncoder().signature() == "ToyHoldemEncoder:v1"
    assert HoldemGame().encoder_signature() == "RawHoldemEncoder:v1"
    assert HoldemGame(encoder=ToyHoldemEncoder()).encoder_signature() == \
        "ToyHoldemEncoder:v1"


def test_toy_merges_raw_states():
    raw, toy = HoldemGame(), HoldemGame(encoder=ToyHoldemEncoder())
    states = _random_decision_states(raw, 400, seed=2)
    raw_keys = {raw.infoset_key(s) for s in states}
    toy_keys = {toy.infoset_key(s) for s in states}
    assert len(toy_keys) < len(raw_keys)
    # And a concrete collision: two different offsuit broadway hands.
    by_toy = {}
    for s in states:
        by_toy.setdefault(toy.infoset_key(s), set()).add(raw.infoset_key(s))
    assert any(len(v) > 1 for v in by_toy.values())


def test_checkpoint_compatibility_matrix(tmp_path):
    raw_solver = MCCFRSolver(HoldemGame(), seed=0)
    raw_solver.train(2)
    raw_path = save_checkpoint(raw_solver, tmp_path / "raw.npz")
    toy_solver = MCCFRSolver(HoldemGame(encoder=ToyHoldemEncoder()), seed=0)
    toy_solver.train(2)
    toy_path = save_checkpoint(toy_solver, tmp_path / "toy.npz")

    load_checkpoint(raw_path, HoldemGame())                              # raw→raw
    load_checkpoint(toy_path, HoldemGame(encoder=ToyHoldemEncoder()))    # toy→toy
    with pytest.raises(CheckpointError, match="encoder"):
        load_checkpoint(raw_path, HoldemGame(encoder=ToyHoldemEncoder()))  # raw→toy
    with pytest.raises(CheckpointError, match="encoder"):
        load_checkpoint(toy_path, HoldemGame())                          # toy→raw


def test_infoset_growth_raw_vs_toy():
    counts = {}
    for name, enc in (("raw", None), ("toy", ToyHoldemEncoder())):
        solver = MCCFRSolver(HoldemGame(encoder=enc), seed=123)
        growth = []
        for _ in range(4):
            solver.train(5)
            growth.append(len(solver.infosets))
        counts[name] = growth
    print(f"infoset growth (every 5 iters, seed 123): {counts}")
    assert counts["toy"][-1] < counts["raw"][-1]
    # Raw keys keep growing roughly linearly (every deal is new); toy keys
    # saturate much sooner.
    assert counts["raw"][-1] - counts["raw"][0] > \
        counts["toy"][-1] - counts["toy"][0]
