"""Native MCCFR parity on a shared random tape (Phases 17, 21, 23, 24).

Both backends consume the identical sequence of uniforms (the "random
tape"): dealing picks ``floor(u * pool)`` with swap-with-last removal,
opponent sampling uses the cdf + searchsorted-right rule. With the same
tape, the two implementations must produce *bitwise identical* training
state — same infosets, same action order, same regret sums, same strategy
sums, same per-iteration values, same number of uniforms consumed.

A single-traversal trace diff (Phase 23) localizes any failure to the first
differing node.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_alpha.solver_config import RELEASE_CONFIG
from poker_alpha.solvers.mccfr import MCCFRSolver

from native_helpers import TapeRng, make_tape

pytest.importorskip("poker_alpha_native")

UNIFORMS_PER_ITERATION = 4000  # generous upper bound for tape sizing


def python_tape_solver(config, tape):
    game = config.build_game()
    solver = MCCFRSolver(game, seed=0)
    solver.rng = TapeRng(tape)
    return solver


def native_tape_solver(config, tape):
    from poker_alpha.native import NativeMCCFRSolver

    return NativeMCCFRSolver(config, seed=0, _tape=tape)


def python_trace(solver, update_player):
    """Mirror of the native trace scheme: opponent nodes pre-order,
    update nodes post-order."""
    game = solver.game
    trace = []

    def walk(state):
        if game.is_terminal(state):
            u0 = game.utility(state)
            u = u0 if update_player == 0 else -u0
            trace.append({"type": "terminal", "utility": u})
            return u
        if game.is_chance(state):
            trace.append({"type": "chance"})
            return walk(game.sample_chance(state, solver.rng))
        player = game.current_player(state)
        key = game.infoset_key(state)
        actions = game.legal_actions(state)
        node = solver._get_infoset(key, actions)
        strategy = node.current_strategy()
        regrets_before = node.regret_sum.copy()
        if player != update_player:
            node.strategy_sum += strategy
            idx = solver._sample(strategy)
            trace.append({"type": "opponent", "player": player, "key": key,
                          "actions": list(actions),
                          "regrets_before": regrets_before.tolist(),
                          "strategy": strategy.tolist(), "sampled": idx})
            return walk(game.next_state(state, actions[idx]))
        child_values = np.zeros(len(actions))
        for i, action in enumerate(actions):
            child_values[i] = walk(game.next_state(state, action))
        from poker_alpha.solvers.cfr import strategy_dot

        node_value = strategy_dot(strategy, child_values)
        node.regret_sum += child_values - node_value
        trace.append({"type": "update", "player": player, "key": key,
                      "actions": list(actions),
                      "regrets_before": regrets_before.tolist(),
                      "strategy": strategy.tolist(),
                      "child_values": child_values.tolist(),
                      "node_value": node_value})
        return node_value

    walk(game.root())
    return trace


def assert_states_equal(py_solver, native_solver):
    """Full bitwise comparison of training state."""
    keys_u64, keys, offsets, tokens, regret, strategy, _ = \
        native_solver._state_arrays()
    names = native_solver._token_names
    assert sorted(py_solver.infosets, key=lambda k: k.encode()) == keys
    for i, key in enumerate(keys):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        node = py_solver.infosets[key]
        assert node.actions == [names[t] for t in tokens[lo:hi]], key
        assert node.regret_sum.tolist() == regret[lo:hi].tolist(), key
        assert node.strategy_sum.tolist() == strategy[lo:hi].tolist(), key


@pytest.mark.parametrize("iterations", [1, 2, 5, 10])
def test_single_trace_and_state_parity(iterations):
    tape = make_tape(seed=50 + iterations, length=iterations * UNIFORMS_PER_ITERATION)
    py = python_tape_solver(RELEASE_CONFIG, tape)
    nat = native_tape_solver(RELEASE_CONFIG, tape)
    for t in range(iterations):
        for up in (0, 1):
            pt = python_trace(py, up)
            nt = nat._core.trace_traverse(up)
            assert len(pt) == len(nt), f"trace length differs at iter {t} up {up}"
            for j, (a, b) in enumerate(zip(pt, nt)):
                assert a == dict(b), f"iter {t} up {up} event {j}:\n{a}\nvs\n{b}"
        py.iterations += 1
    assert py.rng.pos == nat._core.tape_pos()
    assert_states_equal(py, nat)


@pytest.mark.parametrize("iterations", [100])
def test_multi_iteration_parity(iterations):
    tape = make_tape(seed=99, length=iterations * UNIFORMS_PER_ITERATION)
    py = python_tape_solver(RELEASE_CONFIG, tape)
    nat = native_tape_solver(RELEASE_CONFIG, tape)
    py_values = [py.iterate() for _ in range(iterations)]
    nat_values = [nat._core.iterate() for _ in range(iterations)]
    assert py_values == nat_values          # exact
    assert py.rng.pos == nat._core.tape_pos()
    assert py.iterations == nat.iterations == iterations
    assert_states_equal(py, nat)


@pytest.mark.slow
def test_multi_iteration_parity_long():
    iterations = 1_000
    tape = make_tape(seed=123, length=iterations * UNIFORMS_PER_ITERATION)
    py = python_tape_solver(RELEASE_CONFIG, tape)
    nat = native_tape_solver(RELEASE_CONFIG, tape)
    py_values = [py.iterate() for _ in range(iterations)]
    nat_values = [nat._core.iterate() for _ in range(iterations)]
    assert py_values == nat_values
    assert py.rng.pos == nat._core.tape_pos()
    assert_states_equal(py, nat)
    # Average strategies therefore agree exactly as well.
    py_avg = py.average_strategy()
    nat_avg = nat.average_strategy()
    assert set(py_avg) == set(nat_avg)
    for key, probs in py_avg.items():
        for action, p in probs.items():
            assert p == nat_avg[key][action], key
