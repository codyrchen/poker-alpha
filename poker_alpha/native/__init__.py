"""Optional native (C++) MCCFR training backend.

The extension module ``poker_alpha_native`` is built from ``cpp/`` and
installed separately (``pip install ./cpp``); everything else in PokerAlpha
works without it. This package is the only supported way to use it:

* :func:`native_available` — is the extension importable and recent enough?
* :class:`~poker_alpha.native.backend.NativeMCCFRSolver` — high-level solver
  (train / metrics / checkpoint / strategy export).
* :func:`~poker_alpha.native.backend.build_solver` — backend-selecting
  constructor (``python`` / ``native`` / ``auto``).

The Python :class:`~poker_alpha.solvers.mccfr.MCCFRSolver` remains the
reference implementation; the native backend is validated against it by the
parity suite (``tests/test_native_*.py``) and is for *training only* —
strategy lookup and the DecisionEngine never require it.
"""

from __future__ import annotations

REQUIRED_SCHEMA = 1


def native_module():
    """The raw extension module, or ``None`` if not installed/compatible."""
    try:
        import poker_alpha_native  # type: ignore[import-not-found]
    except ImportError:
        return None
    if getattr(poker_alpha_native, "BACKEND_SCHEMA", None) != REQUIRED_SCHEMA:
        return None
    return poker_alpha_native


def native_available() -> bool:
    """True if the native MCCFR backend can be used."""
    return native_module() is not None


def native_version() -> "str | None":
    mod = native_module()
    return None if mod is None else str(mod.__version__)


from .backend import (  # noqa: E402  (re-exports; backend guards the import)
    NativeBackendError,
    NativeMCCFRSolver,
    build_solver,
    native_config,
    supports_config,
)

__all__ = [
    "NativeBackendError",
    "NativeMCCFRSolver",
    "build_solver",
    "native_available",
    "native_config",
    "native_module",
    "native_version",
    "supports_config",
]
