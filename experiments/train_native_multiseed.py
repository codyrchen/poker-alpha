"""Train independent native MCCFR seeds in parallel processes (Phase 45).

External-sampling MCCFR with shared regret tables does not parallelize
safely without changing update semantics, so the clean use of multiple
cores is one solver seed per process. Each worker runs
``experiments/holdem_mccfr_validation.py --backend native`` with its own
checkpoint directory and JSONL log; this wrapper schedules them, bounds the
worker count, resumes existing checkpoints, and reports aggregate
throughput.

Example::

    python experiments/train_native_multiseed.py \
        --seeds 0,1,2 --milestones 100000,200000,300000 \
        --out-dir results/native_training --jobs 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def existing_resume(ckpt_dir: Path, tag: str, seed: int) -> "Path | None":
    """Highest-iteration native checkpoint for this seed, if any."""
    best, best_it = None, -1
    for p in ckpt_dir.glob(f"{tag}_seed{seed}_it*.npz"):
        try:
            it = int(p.stem.split("_it")[-1])
        except ValueError:
            continue
        if it > best_it:
            best, best_it = p, it
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--milestones", default="100000,200000,300000")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel workers (default: min(seeds, cores-2, 4))")
    ap.add_argument("--config", choices=("v2", "legal", "locked"), default="v2")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--resume", action="store_true", default=True,
                    help="resume from existing native checkpoints (default)")
    ap.add_argument("--fresh", dest="resume", action="store_false")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = args.jobs or max(1, min(len(seeds), (os.cpu_count() or 4) - 2, 4))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_flag = {"v2": "--v2-config", "legal": "--legal-sizing-config",
                "locked": "--locked-config"}[args.config]
    tag = {"v2": "v2_native", "legal": "legal_native",
           "locked": "locked_native"}[args.config]

    procs: dict[int, subprocess.Popen] = {}
    pending = list(seeds)
    t0 = time.time()
    results = {}
    logs = {}

    def launch(seed: int) -> subprocess.Popen:
        ckpt_dir = args.out_dir / f"seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{tag}_seed{seed}.jsonl"
        cmd = [sys.executable,
               str(ROOT / "experiments" / "holdem_mccfr_validation.py"),
               cfg_flag, "--backend", "native", "--seed", str(seed),
               "--milestones", args.milestones, "--top-n", str(args.top_n),
               "--ckpt-dir", str(ckpt_dir), "--out", str(out)]
        if args.resume:
            resume = existing_resume(ckpt_dir, tag, seed)
            if resume is not None:
                cmd += ["--resume", str(resume)]
        log = open(args.out_dir / f"seed{seed}.log", "a")
        logs[seed] = log
        print(f"[multiseed] seed {seed}: {' '.join(cmd[1:])}", flush=True)
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        while pending or procs:
            while pending and len(procs) < jobs:
                seed = pending.pop(0)
                procs[seed] = launch(seed)
            time.sleep(5)
            for seed, proc in list(procs.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                results[seed] = rc
                logs[seed].close()
                del procs[seed]
                print(f"[multiseed] seed {seed} exited rc={rc} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
    except KeyboardInterrupt:
        print("[multiseed] interrupt: terminating workers "
              "(native checkpoints are atomic)", flush=True)
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            proc.wait(timeout=60)
        raise SystemExit(130)

    summary = {"seeds": seeds, "jobs": jobs, "milestones": args.milestones,
               "config": args.config, "elapsed_seconds": time.time() - t0,
               "exit_codes": results}
    (args.out_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if any(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
