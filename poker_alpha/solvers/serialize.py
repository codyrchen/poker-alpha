"""Safe, versioned checkpoints for CFR, CFR+ and MCCFR solvers.

Format
------
A checkpoint is an uncompressed NumPy ``.npz`` archive written and read with
``allow_pickle=False`` — no Python objects are ever unpickled. Arrays:

==================  ========================================================
``format_version``  int64 scalar, currently :data:`CHECKPOINT_FORMAT_VERSION`
``solver_type``     unicode scalar: ``"CFR"``, ``"CFR+"`` or ``"MCCFR"``
``game_signature``  unicode scalar, from :meth:`Game.signature`
``encoder_sig``     unicode scalar, :meth:`Game.encoder_signature` or ``""``
``iterations``      int64 scalar
``keys``            unicode array of infoset keys, sorted by UTF-8 bytes
``action_offsets``  int64 array (len = n_infosets + 1) into ``actions``
``actions``         unicode array: each infoset's actions in solver order
``regret_sum``      float64, concatenated per infoset like ``actions``
``strategy_sum``    float64, same layout
``rng_state``       unicode scalar: JSON of ``Generator.bit_generator.state``
                    (MCCFR only; ``""`` otherwise)
==================  ========================================================

Loading verifies the format version, solver type, game signature and encoder
signature and raises :class:`CheckpointError` on any mismatch, so a
checkpoint can never silently resume against a different game or
abstraction. Resuming is exact: ``train(a) + save + load + train(b)`` is
bit-identical to ``train(a + b)`` (including the MCCFR random stream).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Type, Union

import numpy as np

from ..games.base import Game
from .cfr import CFRSolver, InfoSet
from .cfr_plus import CFRPlusSolver
from .mccfr import MCCFRSolver

CHECKPOINT_FORMAT_VERSION = 1

_SOLVER_TYPES: Dict[str, Type[CFRSolver]] = {
    "CFR": CFRSolver,
    "CFR+": CFRPlusSolver,
    "MCCFR": MCCFRSolver,
}

PathLike = Union[str, os.PathLike]


class CheckpointError(Exception):
    """A checkpoint is malformed, unsupported, or incompatible."""


def _solver_type_name(solver: CFRSolver) -> str:
    # Most-derived match first: CFRPlusSolver and MCCFRSolver subclass CFR.
    for name in ("MCCFR", "CFR+", "CFR"):
        if type(solver) is _SOLVER_TYPES[name]:
            return name
    raise CheckpointError(
        f"unsupported solver type {type(solver).__name__}; supported: "
        f"{sorted(_SOLVER_TYPES)}")


def save_checkpoint(solver: CFRSolver, path: PathLike) -> Path:
    """Write ``solver``'s complete training state to ``path`` (``.npz``).

    The write is atomic (temporary file + rename). Returns the final path.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    solver_type = _solver_type_name(solver)
    keys = sorted(solver.infosets, key=lambda k: k.encode("utf-8"))
    offsets = [0]
    actions, regrets, strategies = [], [], []
    for key in keys:
        node = solver.infosets[key]
        actions.extend(node.actions)
        regrets.append(node.regret_sum)
        strategies.append(node.strategy_sum)
        offsets.append(offsets[-1] + len(node.actions))
    rng_state = ""
    if solver_type == "MCCFR":
        rng_state = json.dumps(solver.rng.bit_generator.state, sort_keys=True)
    enc = solver.game.encoder_signature() or ""

    def _ustr(text: str) -> np.ndarray:
        return np.array(text, dtype=np.str_)

    arrays = {
        "format_version": np.array(CHECKPOINT_FORMAT_VERSION, dtype=np.int64),
        "solver_type": _ustr(solver_type),
        "game_signature": _ustr(solver.game.signature()),
        "encoder_sig": _ustr(enc),
        "iterations": np.array(solver.iterations, dtype=np.int64),
        "keys": np.array(keys, dtype=np.str_) if keys
        else np.zeros(0, dtype="<U1"),
        "action_offsets": np.array(offsets, dtype=np.int64),
        "actions": np.array(actions, dtype=np.str_) if actions
        else np.zeros(0, dtype="<U1"),
        "regret_sum": np.concatenate(regrets) if regrets
        else np.zeros(0, dtype=np.float64),
        "strategy_sum": np.concatenate(strategies) if strategies
        else np.zeros(0, dtype=np.float64),
        "rng_state": _ustr(rng_state),
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    return path


def _scalar_str(data, name: str) -> str:
    return str(data[name][()])


def load_checkpoint(path: PathLike, game: Game) -> CFRSolver:
    """Rebuild a solver from ``path`` for ``game``, ready to keep training.

    Raises :class:`CheckpointError` if the file is not a valid checkpoint, its
    version is unsupported, or its game/encoder signature differs from
    ``game``'s.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc

    required = {"format_version", "solver_type", "game_signature",
                "encoder_sig", "iterations", "keys", "action_offsets",
                "actions", "regret_sum", "strategy_sum", "rng_state"}
    missing = required - set(data)
    if missing:
        raise CheckpointError(f"checkpoint missing fields: {sorted(missing)}")
    version = int(data["format_version"])
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format version {version} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})")
    solver_type = _scalar_str(data, "solver_type")
    if solver_type not in _SOLVER_TYPES:
        raise CheckpointError(f"unknown solver type {solver_type!r}")
    stored_game = _scalar_str(data, "game_signature")
    if stored_game != game.signature():
        raise CheckpointError(
            f"game signature mismatch: checkpoint {stored_game!r} vs "
            f"game {game.signature()!r}")
    stored_enc = _scalar_str(data, "encoder_sig")
    game_enc = game.encoder_signature() or ""
    if stored_enc != game_enc:
        raise CheckpointError(
            f"encoder signature mismatch: checkpoint {stored_enc!r} vs "
            f"game {game_enc!r}")

    keys = [str(k) for k in data["keys"]]
    offsets = data["action_offsets"]
    actions = [str(a) for a in data["actions"]]
    regret, strat = data["regret_sum"], data["strategy_sum"]
    if (len(offsets) != len(keys) + 1 or offsets[0] != 0
            or int(offsets[-1]) != len(actions)
            or len(regret) != len(actions) or len(strat) != len(actions)
            or np.any(np.diff(offsets) <= 0)):
        raise CheckpointError("inconsistent checkpoint array layout")

    if solver_type == "MCCFR":
        solver = MCCFRSolver(game, seed=0)
        rng_json = _scalar_str(data, "rng_state")
        if not rng_json:
            raise CheckpointError("MCCFR checkpoint lacks RNG state")
        try:
            state = json.loads(rng_json)
            bitgen_cls = getattr(np.random, state["bit_generator"])
            bitgen = bitgen_cls()
            bitgen.state = state
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            raise CheckpointError(f"bad MCCFR RNG state: {exc}") from exc
        solver.rng = np.random.Generator(bitgen)
    else:
        solver = _SOLVER_TYPES[solver_type](game)
    solver.iterations = int(data["iterations"])
    for i, key in enumerate(keys):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        node = InfoSet(actions[lo:hi])
        node.regret_sum = np.array(regret[lo:hi], dtype=np.float64)
        node.strategy_sum = np.array(strat[lo:hi], dtype=np.float64)
        solver.infosets[key] = node
    return solver
