"""Experiment orchestration (Phase II, Program 13).

Small, dependency-free infrastructure for research runs:

* **content hash**: ``sha256`` over (experiment kind, canonical JSON config,
  SHA-256 of every input artifact, code schema version, seed). Two runs
  with the same hash are the same deterministic computation.
* **cache**: results stored at ``results/experiments/cache/<hash>.json``;
  a rerun of an identical deterministic task returns the cached result.
* **seed grids, parallel workers, per-task timeouts, resume**: tasks run in
  a process pool; a task that exceeds its timeout or raises is recorded
  (config, exception text, partial runtime) and the grid continues;
  rerunning resumes from the cache.
* **registry**: ``results/experiments/index.json`` lists every recorded
  run: name, status, commit, inputs (with hashes), outputs, seeds, wall
  time, peak RSS where available, headline metric and result
  classification. Discovery runs never update the claims registry; see
  :func:`promote_claim`.

CLI::

    python -m poker_alpha.research.runner list
    python -m poker_alpha.research.runner show <name>
    python -m poker_alpha.research.runner verify      # inputs/outputs still match recorded hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "results" / "experiments"
CACHE_DIR = EXP_DIR / "cache"
INDEX = EXP_DIR / "index.json"
SCHEMA_VERSION = 1
CLASSIFICATIONS = ("POSITIVE", "NULL", "NEGATIVE", "INCONCLUSIVE", "DESCRIPTIVE", "VALIDATION")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(kind: str, config: dict, inputs: Sequence[str] = (), seed: Optional[int] = None,
                 schema: int = SCHEMA_VERSION) -> str:
    h = hashlib.sha256()
    h.update(canonical({"kind": kind, "config": config, "schema": schema, "seed": seed}).encode())
    for p in sorted(inputs):
        path = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        h.update(p.encode())
        h.update(file_sha256(path).encode())
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Task:
    kind: str
    config: dict
    seed: Optional[int] = None
    inputs: Sequence[str] = ()
    timeout: Optional[float] = None
    deterministic: bool = True

    @property
    def hash(self) -> str:
        return content_hash(self.kind, self.config, self.inputs, self.seed)


@dataclass
class TaskResult:
    task_hash: str
    status: str                 # "ok" | "cached" | "error" | "timeout"
    result: Any = None
    error: Optional[str] = None
    seconds: float = 0.0
    peak_rss_mb: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _child(fn, config, seed, q):
    t0 = time.perf_counter()
    try:
        res = fn(config, seed)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1 << 20) if sys.platform == "darwin" else rss / (1 << 10)
        q.put(("ok", res, None, time.perf_counter() - t0, rss_mb))
    except BaseException as e:  # noqa: BLE001 - recorded, never hidden
        q.put(("error", None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=8)}",
               time.perf_counter() - t0, None))


def run_task(fn: Callable[[dict, Optional[int]], Any], task: Task, use_cache: bool = True,
             cache_dir: Path = CACHE_DIR) -> TaskResult:
    """Run one task in a child process with an optional timeout; cache
    deterministic successes by content hash."""
    h = task.hash
    cpath = cache_dir / f"{h}.json"
    if use_cache and task.deterministic and cpath.exists():
        d = json.loads(cpath.read_text())
        return TaskResult(h, "cached", d["result"], seconds=d.get("seconds", 0.0), peak_rss_mb=d.get("peak_rss_mb"))
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, task.config, task.seed, q))
    t0 = time.perf_counter()
    p.start()
    p.join(task.timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return TaskResult(h, "timeout", error=f"timeout after {task.timeout} s", seconds=time.perf_counter() - t0)
    try:
        status, res, err, secs, rss = q.get(timeout=5)
    except Exception:  # noqa: BLE001
        return TaskResult(h, "error", error=f"child exited with code {p.exitcode} and no result",
                          seconds=time.perf_counter() - t0)
    tr = TaskResult(h, status, res, err, secs, rss)
    if status == "ok" and task.deterministic and use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"task": {"kind": task.kind, "config": task.config, "seed": task.seed,
                                              "inputs": list(task.inputs)},
                                     "result": res, "seconds": secs, "peak_rss_mb": rss,
                                     "commit": git_commit()}, default=float))
    return tr


def run_grid(fn, kind: str, config: dict, seeds: Sequence[int], inputs: Sequence[str] = (),
             timeout: Optional[float] = None, workers: int = 1, deterministic: bool = True,
             use_cache: bool = True) -> List[TaskResult]:
    """Seed grid; tasks run concurrently up to ``workers`` (each in its own
    process, so a crash or timeout cannot take the grid down)."""
    tasks = [Task(kind, config, s, inputs, timeout, deterministic) for s in seeds]
    if workers <= 1:
        return [run_task(fn, t, use_cache) for t in tasks]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: run_task(fn, t, use_cache), tasks))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def load_index(path: Path = INDEX) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"format": "pokeralpha.experiments.index/v1", "runs": {}}


def record(name: str, *, status: str, outputs: Sequence[str], inputs: Sequence[str] = (), seeds: Any = None,
           wall_seconds: Optional[float] = None, peak_rss_mb: Optional[float] = None, headline: Any = None,
           classification: Optional[str] = None, command: Optional[str] = None, mode: str = "discovery",
           notes: str = "", path: Path = INDEX) -> dict:
    if classification is not None and classification not in CLASSIFICATIONS:
        raise ValueError(f"classification must be one of {CLASSIFICATIONS}")
    if mode not in ("discovery", "confirmation", "validation", "descriptive"):
        raise ValueError("bad mode")
    idx = load_index(path)

    def h(p):
        q = ROOT / p
        return file_sha256(q) if q.exists() else None
    entry = {"name": name, "status": status, "commit": git_commit(), "mode": mode,
             "inputs": {p: h(p) for p in inputs}, "outputs": {p: h(p) for p in outputs},
             "seeds": seeds, "wall_seconds": wall_seconds, "peak_rss_mb": peak_rss_mb,
             "headline": headline, "classification": classification, "command": command, "notes": notes,
             "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    idx["runs"][name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx, indent=1, sort_keys=True, default=float))
    return entry


def record_failure(name: str, config: dict, error: str, partial_seconds: float, path: Path = INDEX) -> dict:
    return record(name, status="failed", outputs=[], seeds=None, wall_seconds=partial_seconds,
                  notes=json.dumps({"config": config, "error": error[:4000]}, default=str), path=path)


def verify(path: Path = INDEX) -> List[str]:
    """Outputs/inputs whose current SHA-256 differs from the recorded one."""
    problems = []
    for name, e in load_index(path)["runs"].items():
        for kind in ("inputs", "outputs"):
            for p, sha in e.get(kind, {}).items():
                q = ROOT / p
                if sha is None:
                    continue
                if not q.exists():
                    problems.append(f"{name}: {kind[:-1]} missing: {p}")
                elif file_sha256(q) != sha:
                    problems.append(f"{name}: {kind[:-1]} changed since recording: {p}")
    return problems


def promote_claim(claim_id: str, value: Any, units: str, source: str, qualifier: str, run_name: str,
                  index_path: Path = INDEX, claims_path: Path = ROOT / "results" / "claims.json") -> dict:
    """Add a claim ONLY from a recorded *confirmation* run with a POSITIVE
    or VALIDATION classification. Discovery results can never be claimed."""
    run = load_index(index_path)["runs"].get(run_name)
    if run is None:
        raise ValueError(f"unknown run {run_name!r}")
    if run["mode"] not in ("confirmation", "validation") or run["classification"] not in ("POSITIVE", "VALIDATION"):
        raise ValueError("only confirmed POSITIVE / VALIDATION runs can back a public claim")
    claims = json.loads(claims_path.read_text())
    claims["claims"][claim_id] = {"value": value, "units": units, "source": source, "qualifier": qualifier,
                                  "experiment": run_name}
    claims_path.write_text(json.dumps(claims, indent=1))
    return claims["claims"][claim_id]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m poker_alpha.research.runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("name")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    idx = load_index()
    if a.cmd == "list":
        for name, e in sorted(idx["runs"].items()):
            print(f"{name:40s} {e['status']:8s} {e.get('mode', ''):12s} {str(e.get('classification')):12s} "
                  f"{(e.get('commit') or '')[:8]}  {json.dumps(e.get('headline'), default=str)[:70]}")
    elif a.cmd == "show":
        print(json.dumps(idx["runs"].get(a.name), indent=1))
    elif a.cmd == "verify":
        probs = verify()
        print("\n".join(probs) if probs else "all recorded inputs/outputs match")
        return 1 if probs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
