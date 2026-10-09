"""One-command native solver validation suite (Phase 82).

Runs the important native correctness checks and prints a PASS/FAIL table:

1. game mechanics + encoder parity on a random reachable-state corpus;
2. evaluator parity (random five/six/seven-card sample);
3. feature parity (strength/draw/nut/blocker/texture/river percentile);
4. random-tape MCCFR parity (bitwise, multi-iteration);
5. checkpoint save/load/exact-resume;
6. strategy artifact export -> Python loader -> DecisionEngine provider.

This wraps the same assertions as the pytest suite (tests/test_native_*)
in a single reproducible command::

    python experiments/validate_native_solver.py [--hands 400] [--iters 100]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=400,
                    help="random hands for the state-parity corpus")
    ap.add_argument("--iters", type=int, default=100,
                    help="random-tape parity iterations")
    args = ap.parse_args()

    from poker_alpha.native import native_available, native_version

    if not native_available():
        print("FAIL native backend not installed (pip install ./cpp)")
        return 1
    print(f"native backend {native_version()} — validation suite")

    checks = []

    def run(name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            checks.append((name, True, detail, time.perf_counter() - t0))
        except Exception:  # noqa: BLE001 — report and continue
            checks.append((name, False, traceback.format_exc(limit=4),
                           time.perf_counter() - t0))

    def game_parity():
        from test_native_game_parity import _run_corpus
        from poker_alpha.solver_config import RELEASE_CONFIG

        n = _run_corpus(RELEASE_CONFIG, args.hands, seed=7)
        return f"{n} states compared (mechanics, legal actions, keys, utility)"

    def evaluator_parity():
        import numpy as np
        from test_native_evaluator import _check

        rng = np.random.default_rng(0)
        for size in (5, 6, 7):
            for _ in range(5_000):
                _check(rng.choice(52, size=size, replace=False).tolist())
        return "15,000 random hands (5/6/7 cards)"

    def feature_parity():
        from test_native_features import _assert_match, _random_cases

        n = 0
        for board_n, cnt in ((3, 500), (4, 500), (5, 300)):
            for hole, board in _random_cases(cnt, board_n, seed=30 + board_n):
                _assert_match(hole, board)
                n += 1
        return f"{n} feature vectors"

    def tape_parity():
        from test_native_mccfr_parity import (
            UNIFORMS_PER_ITERATION, assert_states_equal, make_tape,
            native_tape_solver, python_tape_solver)
        from poker_alpha.solver_config import RELEASE_CONFIG

        tape = make_tape(seed=99, length=args.iters * UNIFORMS_PER_ITERATION)
        py = python_tape_solver(RELEASE_CONFIG, tape)
        nat = native_tape_solver(RELEASE_CONFIG, tape)
        pv = [py.iterate() for _ in range(args.iters)]
        nv = [nat._core.iterate() for _ in range(args.iters)]
        assert pv == nv
        assert py.rng.pos == nat._core.tape_pos()
        assert_states_equal(py, nat)
        return f"{args.iters} iterations bitwise-identical"

    def checkpoint_resume():
        from poker_alpha.native import NativeMCCFRSolver
        from poker_alpha.solver_config import RELEASE_CONFIG

        with tempfile.TemporaryDirectory() as td:
            full = NativeMCCFRSolver(RELEASE_CONFIG, seed=3)
            full.train(200)
            part = NativeMCCFRSolver(RELEASE_CONFIG, seed=3)
            part.train(120)
            path = part.save_checkpoint(Path(td) / "c.npz")
            resumed = NativeMCCFRSolver.load_checkpoint(path, RELEASE_CONFIG)
            resumed.train(80)
            a = full._state_arrays()
            b = resumed._state_arrays()
            assert a[1] == b[1]
            assert (a[4] == b[4]).all() and (a[5] == b[5]).all()
            assert (a[6] == b[6]).all()
        return "train(120)+save+load+train(80) == train(200) exactly"

    def artifact_roundtrip():
        from poker_alpha.native import NativeMCCFRSolver
        from poker_alpha.solver_config import RELEASE_CONFIG
        from poker_alpha.solvers.strategy_artifact import load_artifact
        from poker_alpha.decision.strategy import SolverStrategyProvider

        with tempfile.TemporaryDirectory() as td:
            s = NativeMCCFRSolver(RELEASE_CONFIG, seed=8)
            s.train(200)
            path = s.export_strategy(Path(td) / "a.npz")
            art = load_artifact(path, RELEASE_CONFIG.build_game())
            assert len(art) > 0 and art.meta["backend"] == "native"
            provider = SolverStrategyProvider.from_artifact(
                path, config=RELEASE_CONFIG)
            assert hasattr(provider, "lookup")
        return "export -> load_artifact -> SolverStrategyProvider"

    run("game/encoder parity", game_parity)
    run("evaluator parity", evaluator_parity)
    run("feature parity", feature_parity)
    run("random-tape MCCFR parity", tape_parity)
    run("checkpoint exact resume", checkpoint_resume)
    run("artifact roundtrip", artifact_roundtrip)

    width = max(len(n) for n, *_ in checks)
    failures = 0
    for name, ok, detail, dt in checks:
        status = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"  [{status}] {name:<{width}}  ({dt:5.1f}s)  {detail if ok else ''}")
        if not ok:
            print(detail)
    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
