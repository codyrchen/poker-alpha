"""Phase II (Programs 11 / 59-62): targeted extension of the release lineage
from 2M to 3M and 4M iterations for seeds 0-2, to answer two measured
questions (NOT to promote a release):

  Q1 (seed disagreement) - does mean pairwise seed disagreement shrink like
     sampling noise (2M->4M ratio ~0.71) or stay (structural)?
  Q2 (red-team loop) - do the confirmed local-sensitivity failure states,
     concentrated on low-visit keys, become less sensitive with training?

Resumes results/native_training/seed{s}/v2_native_seed{s}_it2000000.npz
(bit-exact resume), exports strategy artifacts to
results/strategy/candidates/phase2_extension/ and saves 4M checkpoints next
to the originals (git-ignored).

    python experiments/phase2_extend_training.py --seed 0      # one process per seed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.native import NativeMCCFRSolver  # noqa: E402
from poker_alpha.solver_config import RELEASE_CONFIG  # noqa: E402

OUT = ROOT / "results" / "strategy" / "candidates" / "phase2_extension"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--milestones", default="3000000,4000000")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = ROOT / "results" / "native_training" / f"seed{a.seed}" / f"v2_native_seed{a.seed}_it2000000.npz"
    s = NativeMCCFRSolver.load_checkpoint(ck, RELEASE_CONFIG)
    assert s.iterations == 2_000_000, s.iterations
    log = []
    for m in map(int, a.milestones.split(",")):
        t = time.time()
        s.train(m - s.iterations)
        dt = time.time() - t
        art = OUT / f"holdem_v2_native_seed{a.seed}_{m // 1000}k.npz"
        s.export_strategy(art, meta={"iterations": m, "lineage": f"native seed {a.seed}: release lineage 2M checkpoint -> {m}",
                                     "purpose": "Phase II research extension; NOT a release candidate"})
        rec = {"seed": a.seed, "iterations": m, "segment_seconds": round(dt, 1), "metrics": s.metrics(), "artifact": str(art.relative_to(ROOT))}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    s.save_checkpoint(ROOT / "results" / "native_training" / f"seed{a.seed}" / f"v2_native_seed{a.seed}_it{s.iterations}.npz")
    (OUT / f"seed{a.seed}_log.json").write_text(json.dumps(log, indent=1, default=float))


if __name__ == "__main__":
    main()
