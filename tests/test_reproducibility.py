"""Reproducibility safety net: pin exact solver outputs.

Each test trains a solver deterministically and pins a canonical strategy
digest (see :mod:`poker_alpha.solvers.digest`), the resulting game value and
exploitability. Any change to solver arithmetic, traversal order, RNG
consumption, or game rules that alters results will trip these tests. If such
a change is *deliberate*, update the pins in the same commit and document why.

Pins were recorded at the baseline commit 433c52f (before the Hold'em
platform work) and must stay valid unless a migration is documented.

MIGRATION (Phase 25): the original pins were only reproducible on CPUs where
OpenBLAS selects the same ``ddot`` kernel as the recording machine. GitHub
Actions runners (OpenBLAS Haswell/Zen kernels) produced different Leduc
digests — and, because one ULP flipped a sampling decision, a different
seeded Leduc MCCFR trajectory (game value -0.1148 instead of -0.0759).
Solvers now compute strategy-weighted values with an exactly rounded,
platform-independent sum (``solvers.cfr.strategy_dot``). The four Kuhn pins
were unchanged by the migration; the three Leduc pins below were re-recorded
and verified identical under the OpenBLAS SkylakeX, Haswell and Prescott
kernels and with NumPy's AVX2/AVX-512 paths disabled. Previous Leduc
digests: leduc_cfr_20 5390b065..., leduc_cfr_plus_50 48bb5b20...,
leduc_mccfr_seed11_2000 08122b00....
"""

import pytest

from poker_alpha.games import KuhnPoker, LeducPoker
from poker_alpha.solvers import (CFRPlusSolver, CFRSolver, MCCFRSolver,
                                 expected_value, exploitability)
from poker_alpha.solvers.digest import (canonical_strategy_bytes,
                                        strategy_digest)

# (factory, iterations, digest, game value, exploitability)
PINS = {
    "kuhn_cfr_1000": (
        lambda: CFRSolver(KuhnPoker()), KuhnPoker, 1000,
        "5b17b98dd35657e31d09a7e8faff401a4e3eff0af9079bd57fcb5c2fc99ad56a",
        -0.0551776168210788, 0.006504893205705617),
    "kuhn_cfr_plus_1000": (
        lambda: CFRPlusSolver(KuhnPoker()), KuhnPoker, 1000,
        "b52abe5543374a387a392e8a9cc709daf6e79bd20c966eb6d25d0162e0b89e25",
        -0.055555917582651854, 8.736532252073478e-05),
    "kuhn_mccfr_seed7_20000": (
        lambda: MCCFRSolver(KuhnPoker(), seed=7), KuhnPoker, 20_000,
        "542c686993607961e22576201c20f4315736fa193e1e3ddbfdf19079f5229efa",
        -0.0555943747546295, 0.00574898365888217),
    "leduc_cfr_20": (
        lambda: CFRSolver(LeducPoker()), LeducPoker, 20,
        "af63936dd575123bec4522725575ca5cb396176941f41700252db93636663be6",
        -0.11574584187152509, 0.19314248478612467),
    "leduc_mccfr_seed11_2000": (
        lambda: MCCFRSolver(LeducPoker(), seed=11), LeducPoker, 2000,
        "ddb1f2560a2a53539a4b735c249626b80db099c9696fcfe8406cd8328d948769",
        -0.11483613315775798, 0.49581637915174925),
}

SLOW_PINS = {
    "kuhn_cfr_20000": (
        lambda: CFRSolver(KuhnPoker()), KuhnPoker, 20_000,
        "43e1c152303b6651abce9d19c67c34d4ee04057ab16153b0f2a111b1f9919a52",
        -0.055555107992281305, 0.0016078547525372779),
    "leduc_cfr_plus_50": (
        lambda: CFRPlusSolver(LeducPoker()), LeducPoker, 50,
        "cf14c9d2acd9307e0d995239d65ed7977b933c7fa695e5e3fa6add8176cb2e02",
        -0.08623878844551527, 0.034121456494549535),
}

TOL = 1e-12


def _check(pin):
    factory, game_cls, iterations, digest, value, expl = pin
    solver = factory()
    solver.train(iterations)
    strategy = solver.average_strategy()
    game = game_cls()
    assert strategy_digest(strategy) == digest
    assert expected_value(game, strategy) == pytest.approx(value, abs=TOL)
    assert exploitability(game, strategy) == pytest.approx(expl, abs=TOL)


@pytest.mark.parametrize("name", sorted(PINS))
def test_pinned_solver_outputs(name):
    _check(PINS[name])


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SLOW_PINS))
def test_pinned_solver_outputs_slow(name):
    _check(SLOW_PINS[name])


# -- digest properties -------------------------------------------------------

def test_strategy_dot_is_exactly_rounded_and_platform_independent():
    """Guards the migration: strategy-weighted values must not depend on the
    BLAS kernel or the reduction order."""
    import math

    import numpy as np

    from poker_alpha.solvers.cfr import strategy_dot

    s = np.array([0.1, 0.2, 0.7])
    v = np.array([1e16, 1.0, -1e16])
    exact = math.fsum([0.1 * 1e16, 0.2 * 1.0, 0.7 * -1e16])
    assert strategy_dot(s, v) == exact
    assert strategy_dot(s[::-1].copy(), v[::-1].copy()) == exact   # order-free


def test_digest_independent_of_dict_order():
    a = {"x": {"p": 0.25, "b": 0.75}, "y": {"c": 1.0}}
    b = {"y": {"c": 1.0}, "x": {"b": 0.75, "p": 0.25}}
    assert strategy_digest(a) == strategy_digest(b)
    assert strategy_digest(a, None) == strategy_digest(b, None)


def test_digest_normalizes_negative_zero():
    assert strategy_digest({"k": {"a": -0.0, "b": 1.0}}, None) == \
        strategy_digest({"k": {"a": 0.0, "b": 1.0}}, None)


def test_digest_detects_changes():
    base = {"k": {"a": 0.5, "b": 0.5}}
    assert strategy_digest(base) != strategy_digest({"k": {"a": 0.5, "b": 0.5 + 1e-9}})
    assert strategy_digest(base) != strategy_digest({"k2": {"a": 0.5, "b": 0.5}})
    assert strategy_digest(base) != strategy_digest({"k": {"a": 0.5, "c": 0.5}})


def test_digest_quantization_absorbs_ulp_noise():
    p = 1.0 / 3.0
    assert strategy_digest({"k": {"a": p}}) == \
        strategy_digest({"k": {"a": p + 2e-16}})


def test_digest_rejects_nan():
    with pytest.raises(ValueError):
        strategy_digest({"k": {"a": float("nan")}})


def test_canonical_bytes_are_explicit_and_length_prefixed():
    raw = canonical_strategy_bytes({"ab": {"c": 1.0}})
    assert raw.startswith(b"PASD\x01")
    # Key/action boundaries are length-prefixed, so these must differ.
    assert canonical_strategy_bytes({"a": {"bc": 1.0}}) != raw
