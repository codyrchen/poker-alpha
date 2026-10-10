"""High-level wrapper around the native MCCFR extension.

Design (docs/native_solver_design.md): the C++ core owns the hot path; this
module owns configuration transport, file formats and backend selection.
Checkpoints and strategy artifacts are written *here*, in Python, from bulk
arrays the core hands over in one call — the native side has no file I/O.

Native checkpoint format (``pokeralpha.native_checkpoint/v1``): uncompressed
``.npz``, ``allow_pickle=False``, holding the config/game/encoder signatures,
backend schema + RNG name, iteration count, the packed uint64 keys *and*
their canonical rendered strings, per-infoset action tokens, float64
regret/strategy sums, the xoshiro256** state and a content SHA-256. Loading
validates all of it and raises :class:`NativeBackendError` on any mismatch;
resume is exact (bit-identical to an uninterrupted run).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from ..solver_config import HoldemSolverConfig

CHECKPOINT_FORMAT = "pokeralpha.native_checkpoint/v1"

# Compact abstract-history encoders the native port implements, mapped to
# their river percentile bucket count.
_SUPPORTED_ENCODERS = {
    "compact": 0,
    "compact_river_pct10": 10,
    "compact_river_pct20": 20,
}


class NativeBackendError(Exception):
    """Native backend unavailable, unsupported config, or bad checkpoint."""


def _require_module():
    from . import native_module

    mod = native_module()
    if mod is None:
        raise NativeBackendError(
            "native backend not installed (pip install ./cpp) or its schema "
            "does not match this PokerAlpha version")
    return mod


def supports_config(config: HoldemSolverConfig) -> bool:
    """Does the native backend implement this solver configuration?"""
    return config.encoder in _SUPPORTED_ENCODERS


def native_config(config: HoldemSolverConfig):
    """Translate a HoldemSolverConfig into the extension's NativeConfig."""
    mod = _require_module()
    if not supports_config(config):
        raise NativeBackendError(
            f"native backend does not support encoder {config.encoder!r} "
            f"(supported: {sorted(_SUPPORTED_ENCODERS)})")
    game = config.build_game()
    return mod.NativeConfig(
        starting_stack=float(config.starting_stack),
        raise_cap=int(config.raise_cap),
        enforce_min_raise=bool(config.enforce_min_raise),
        postflop=[(name, float(v)) for name, v in config.bet_fractions],
        preflop=[(name, float(v)) for name, v in config.preflop_raise_multiples],
        texture=True,
        river_blockers=True,
        river_pct_buckets=_SUPPORTED_ENCODERS[config.encoder],
        config_signature=config.signature(),
        game_signature=game.signature(),
        encoder_signature=game.encoder_signature() or "",
    )


def _content_hash(keys, offsets, tokens, regret, strategy) -> str:
    h = hashlib.sha256()
    for part in (np.asarray(keys, np.uint64).tobytes(),
                 np.asarray(offsets, np.int64).tobytes(),
                 np.asarray(tokens, np.uint8).tobytes(),
                 np.asarray(regret, np.float64).tobytes(),
                 np.asarray(strategy, np.float64).tobytes()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()


class NativeMCCFRSolver:
    """External-sampling MCCFR on the native backend.

    One coarse pybind call per :meth:`train` chunk; the GIL is released
    while the C++ core runs. Ctrl+C is honoured at chunk boundaries.
    """

    def __init__(self, config: HoldemSolverConfig, seed: int = 0, *,
                 _tape=None) -> None:
        mod = _require_module()
        self.config = config
        self.seed = int(seed)
        self._ncfg = native_config(config)
        if _tape is not None:
            self._core = mod.NativeSolverCore(self._ncfg, list(map(float, _tape)))
        else:
            self._core = mod.NativeSolverCore(self._ncfg, self.seed)
        self._token_names = self._all_token_names(config)

    @staticmethod
    def _all_token_names(config: HoldemSolverConfig):
        names = ["f", "c", "a"]
        names += [name for name, _ in config.bet_fractions]
        names += [name for name, _ in config.preflop_raise_multiples]
        return names

    # -- training ---------------------------------------------------------

    @property
    def iterations(self) -> int:
        return int(self._core.iterations)

    def train(self, iterations: int, progress_every: int = 0,
              progress: Optional[Callable[[dict], None]] = None,
              chunk: int = 2000) -> None:
        """Train ``iterations`` iterations (resumable, interruptible)."""
        remaining = int(iterations)
        step = progress_every if (progress_every and progress) else remaining
        while remaining > 0:
            take = min(step, remaining) if step else remaining
            self._core.train(take, chunk)
            remaining -= take
            if progress is not None and progress_every:
                progress(self.metrics())

    def metrics(self) -> dict:
        return dict(self._core.metrics())

    # -- strategy access ---------------------------------------------------

    def _state_arrays(self):
        (keys_u64, keys_str, offsets, tokens, regret, strategy,
         rng_state) = self._core.export_state()
        return keys_u64, list(keys_str), offsets, tokens, regret, strategy, rng_state

    def average_strategy(self) -> Dict[str, Dict[str, float]]:
        """``{infoset_key: {action: prob}}`` — export-sized, not a hot path."""
        _, keys, offsets, tokens, _, strategy, _ = self._state_arrays()
        out: Dict[str, Dict[str, float]] = {}
        names = self._token_names
        for i, key in enumerate(keys):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            ssum = strategy[lo:hi]
            total = float(ssum.sum())
            if total > 0:
                probs = ssum / total
            else:
                probs = np.full(hi - lo, 1.0 / (hi - lo))
            out[key] = {names[t]: float(p) for t, p in zip(tokens[lo:hi], probs)}
        return out

    class _ShimNode:
        __slots__ = ("actions", "strategy_sum", "regret_sum")

        def __init__(self, actions, strategy_sum, regret_sum):
            self.actions = actions
            self.strategy_sum = strategy_sum
            self.regret_sum = regret_sum

        def average_strategy(self):
            total = self.strategy_sum.sum()
            if total > 0.0:
                return self.strategy_sum / total
            return np.full(len(self.actions), 1.0 / len(self.actions))

    def _as_python_solver(self):
        """A duck-typed stand-in accepted by strategy_artifact.export_solver."""
        from types import SimpleNamespace

        _, keys, offsets, tokens, regret, strategy, _ = self._state_arrays()
        names = self._token_names
        infosets = {}
        for i, key in enumerate(keys):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            infosets[key] = self._ShimNode(
                [names[t] for t in tokens[lo:hi]],
                np.array(strategy[lo:hi], dtype=np.float64),
                np.array(regret[lo:hi], dtype=np.float64))
        return SimpleNamespace(game=self.config.build_game(),
                               infosets=infosets,
                               iterations=self.iterations)

    def export_strategy(self, path, meta: Optional[dict] = None,
                        min_visits: float = 0.0) -> Path:
        """Write a standard strategy artifact (same format/loader as Python)."""
        from . import native_version
        from ..solvers.strategy_artifact import export_solver

        mod = _require_module()
        m = dict(meta or {})
        m.setdefault("seed", self.seed)
        m["backend"] = "native"
        m["native_backend_version"] = native_version()
        m["native_backend_schema"] = int(mod.BACKEND_SCHEMA)
        m["native_rng"] = str(mod.RNG_NAME)
        m["sampler"] = "NativeMCCFRSolver"
        return export_solver(self._as_python_solver(), path, meta=m,
                             min_visits=min_visits)

    # -- checkpoints --------------------------------------------------------

    def save_checkpoint(self, path) -> Path:
        mod = _require_module()
        keys_u64, keys, offsets, tokens, regret, strategy, rng_state = \
            self._state_arrays()
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        arrays = {
            "format": np.array(CHECKPOINT_FORMAT),
            "backend_schema": np.array(int(mod.BACKEND_SCHEMA), dtype=np.int64),
            "rng_name": np.array(str(mod.RNG_NAME)),
            "solver_config": np.array(self.config.signature()),
            "game_signature": np.array(self._ncfg.game_signature),
            "encoder_sig": np.array(self._ncfg.encoder_signature),
            "seed": np.array(self.seed, dtype=np.int64),
            "iterations": np.array(self.iterations, dtype=np.int64),
            "keys_u64": np.asarray(keys_u64, dtype=np.uint64),
            "keys": (np.array(keys, dtype=np.str_) if keys
                     else np.zeros(0, dtype="<U1")),
            "action_offsets": np.asarray(offsets, dtype=np.int64),
            "action_tokens": np.asarray(tokens, dtype=np.uint8),
            "regret_sum": np.asarray(regret, dtype=np.float64),
            "strategy_sum": np.asarray(strategy, dtype=np.float64),
            "rng_state": np.asarray(rng_state, dtype=np.uint64),
            "content_sha256": np.array(_content_hash(
                keys_u64, offsets, tokens, regret, strategy)),
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
        return path

    @classmethod
    def load_checkpoint(cls, path, config: HoldemSolverConfig,
                        seed: Optional[int] = None) -> "NativeMCCFRSolver":
        mod = _require_module()
        try:
            with np.load(Path(path), allow_pickle=False) as npz:
                data = {k: npz[k] for k in npz.files}
        except Exception as exc:  # noqa: BLE001 — truncated/corrupt/not-a-zip
            raise NativeBackendError(
                f"cannot read native checkpoint {path}: "
                f"{type(exc).__name__}: {exc}") from exc
        required = {"format", "backend_schema", "rng_name", "solver_config",
                    "game_signature", "encoder_sig", "seed", "iterations",
                    "keys_u64", "keys", "action_offsets", "action_tokens",
                    "regret_sum", "strategy_sum", "rng_state",
                    "content_sha256"}
        missing = required - set(data)
        if missing:
            raise NativeBackendError(
                f"native checkpoint missing fields: {sorted(missing)}")
        if str(data["format"][()]) != CHECKPOINT_FORMAT:
            raise NativeBackendError(
                f"not a native checkpoint (format {data['format'][()]!r})")
        if int(data["backend_schema"]) != int(mod.BACKEND_SCHEMA):
            raise NativeBackendError(
                f"backend schema mismatch: checkpoint "
                f"{int(data['backend_schema'])} vs installed {int(mod.BACKEND_SCHEMA)}")
        if str(data["rng_name"][()]) != str(mod.RNG_NAME):
            raise NativeBackendError("RNG algorithm mismatch")
        if str(data["solver_config"][()]) != config.signature():
            raise NativeBackendError(
                f"solver config mismatch: checkpoint "
                f"{data['solver_config'][()]!r} vs {config.signature()!r}")
        game = config.build_game()
        if str(data["game_signature"][()]) != game.signature():
            raise NativeBackendError("game signature mismatch")
        if str(data["encoder_sig"][()]) != (game.encoder_signature() or ""):
            raise NativeBackendError("encoder signature mismatch")
        expect = _content_hash(data["keys_u64"], data["action_offsets"],
                               data["action_tokens"], data["regret_sum"],
                               data["strategy_sum"])
        if str(data["content_sha256"][()]) != expect:
            raise NativeBackendError(
                "native checkpoint content checksum mismatch (corrupt)")
        offsets = data["action_offsets"]
        keys_u64 = data["keys_u64"]
        if (len(offsets) != len(keys_u64) + 1
                or (len(offsets) and int(offsets[0]) != 0)
                or int(offsets[-1]) != len(data["action_tokens"])
                or len(data["regret_sum"]) != len(data["action_tokens"])
                or len(data["strategy_sum"]) != len(data["action_tokens"])
                or len(data["rng_state"]) != 4):
            raise NativeBackendError("inconsistent native checkpoint layout")

        solver = cls(config, seed=int(data["seed"]) if seed is None else seed)
        try:
            solver._core.import_state(
                keys_u64=np.ascontiguousarray(keys_u64, dtype=np.uint64),
                offsets=np.ascontiguousarray(offsets, dtype=np.int64),
                tokens=np.ascontiguousarray(data["action_tokens"], dtype=np.uint8),
                regret=np.ascontiguousarray(data["regret_sum"], dtype=np.float64),
                strategy=np.ascontiguousarray(data["strategy_sum"], dtype=np.float64),
                rng_state=np.ascontiguousarray(data["rng_state"], dtype=np.uint64),
                iterations=int(data["iterations"]))
        except ValueError as exc:
            raise NativeBackendError(f"bad native checkpoint state: {exc}") from exc
        return solver


def build_solver(config: HoldemSolverConfig, seed: int, backend: str = "auto",
                 echo: Optional[Callable[[str], None]] = None):
    """Backend-selecting constructor.

    ``python`` — always the reference :class:`MCCFRSolver`;
    ``native`` — require the native backend (raise if unavailable);
    ``auto`` — native when installed and the config is supported, else Python.
    Returns ``(solver, backend_name)`` and never silently misreports which
    backend is active.
    """
    from . import native_available

    if backend not in ("python", "native", "auto"):
        raise ValueError(f"unknown backend {backend!r}")
    use_native = False
    if backend == "native":
        if not native_available():
            raise NativeBackendError("native backend requested but not installed")
        if not supports_config(config):
            raise NativeBackendError(
                f"native backend does not support encoder {config.encoder!r}")
        use_native = True
    elif backend == "auto":
        use_native = native_available() and supports_config(config)
    if echo is not None:
        echo(f"solver backend: {'native' if use_native else 'python'}")
    if use_native:
        return NativeMCCFRSolver(config, seed=seed), "native"
    return config.build_solver(seed=seed), "python"
