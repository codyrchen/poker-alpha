"""Shared helpers for the native-backend parity tests.

Everything here runs on the *Python* side; the native side is reached only
through the extension's debug API. Tests import this module and skip when
the extension is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_alpha.games.holdem import HoldemState, _tokens
from poker_alpha.solver_config import HoldemSolverConfig

native = pytest.importorskip("poker_alpha_native")


def native_cfg(config: HoldemSolverConfig):
    from poker_alpha.native.backend import native_config

    return native_config(config)


def pack_hand_value(value) -> int:
    """Python hand-value tuple -> the native packed uint32 encoding."""
    cat, *ties = value
    out = cat << 20
    shifts = (16, 12, 8, 4, 0)
    for t, sh in zip(ties, shifts):
        out |= t << sh
    return out


def state_fields(state: HoldemState):
    """(holes flat list, board list, streets token lists) for debug_state_info."""
    holes = [] if state.holes is None else [c for h in state.holes for c in h]
    streets = [list(_tokens(sa)) for sa in state.streets]
    return holes, list(state.board), streets


def native_info(ncfg, state: HoldemState):
    holes, board, streets = state_fields(state)
    return native.debug_state_info(ncfg, holes, board, streets)


class TapeRng:
    """Deterministic stand-in for numpy's Generator, driven by a tape of
    uniforms. Implements exactly the subset MCCFR + HoldemGame use, with the
    documented tape rules (floor(u * size), swap-with-last removal) — the
    same rules the native RandomSource implements in tape mode.
    """

    def __init__(self, tape):
        self.tape = list(tape)
        self.pos = 0

    def _next(self) -> float:
        if self.pos >= len(self.tape):
            raise RuntimeError("random tape exhausted")
        u = self.tape[self.pos]
        self.pos += 1
        return u

    def random(self) -> float:
        return self._next()

    def choice(self, n, size=None, replace=True, p=None):
        assert p is None and replace is False and size is not None, \
            "TapeRng implements only choice(n, size=k, replace=False)"
        pool = list(range(int(n)))
        picked = []
        for _ in range(int(size)):
            u = self._next()
            i = int(u * len(pool))
            if i >= len(pool):
                i = len(pool) - 1
            picked.append(pool[i])
            pool[i] = pool[-1]
            pool.pop()
        return np.array(picked, dtype=np.int64)


def make_tape(seed: int, length: int):
    """A reproducible tape of uniforms (generator identity is irrelevant —
    both backends consume the same numbers)."""
    rng = np.random.default_rng(seed)
    return rng.random(length).tolist()


def random_state_corpus(config: HoldemSolverConfig, n_hands: int, seed: int,
                        include_chance: bool = True):
    """Reachable states from random legal play of the *Python* game.

    Yields every state along each trajectory (decision, chance and terminal
    states), which is what the mechanics-parity tests compare.
    """
    game = config.build_game()
    rng = np.random.default_rng(seed)
    for _ in range(n_hands):
        state = game.root()
        yield game, state
        while True:
            if game.is_terminal(state):
                break
            if game.is_chance(state):
                state = game.sample_chance(state, rng)
            else:
                actions = game.legal_actions(state)
                state = game.next_state(state, actions[int(rng.integers(len(actions)))])
            yield game, state
