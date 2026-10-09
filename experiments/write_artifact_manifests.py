"""Write sidecar manifests for committed v1 strategy artifacts (Phase 55).

Non-destructive: the .npz files are not rewritten. Each manifest records the
file's SHA-256 (verified by load_artifact from then on), its signatures, the
full config of the signature, seed / iterations / sampler from the artifact
meta, and provenance. The generation commit was not recorded when these
artifacts were made, so it is null; ``first_committed_in`` is the commit that
added the file to Git.

    python experiments/write_artifact_manifests.py results/strategy/holdem_v2_seed0.npz ...
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.solver_config import config_for_signature  # noqa: E402
from poker_alpha.solvers.strategy_artifact import MANIFEST_FORMAT, manifest_path  # noqa: E402
from poker_alpha.utils.provenance import file_sha256  # noqa: E402


def first_commit(path: Path):
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%H",
                          "--", str(path.relative_to(ROOT))], capture_output=True, text=True)
    lines = out.stdout.split()
    return lines[-1] if lines else None


def main(paths):
    rc = json.loads((ROOT / "results/validation/release_candidate.json").read_text())
    for p in map(Path, paths):
        p = p.resolve()
        with np.load(p, allow_pickle=False) as z:
            fmt = str(z["format"][()])
            sig = str(z["config_sig"][()])
            game_sig = str(z["game_signature"][()])
            enc = str(z["encoder_sig"][()])
            meta = json.loads(str(z["meta"][()]))
            n = int(len(z["keys"]))
        cfg = config_for_signature(sig)
        if cfg is None:
            raise SystemExit(f"{p}: unknown config signature {sig}")
        c = cfg.to_dict()
        art_rc = rc["solver"]["artifact"]
        gen_cmd = art_rc["generation_command"] if art_rc["path"] == str(p.relative_to(ROOT)) \
            else meta.get("generation_command")
        m = {"format": MANIFEST_FORMAT, "artifact": p.name, "artifact_format": fmt,
             "sha256": file_sha256(p), "bytes": p.stat().st_size, "infosets": n,
             "config_signature": sig, "config": c,
             "config_version": c["config_version"], "game_signature": game_sig,
             "encoder_signature": enc, "action_abstraction": c.get("action_abstraction",
                                                                  c.get("bet_fractions")),
             "starting_stack": c["starting_stack"], "blinds": [0.5, 1.0],
             "raise_cap": c["raise_cap"], "averaging": c.get("averaging", "uniform"),
             "sampler": meta.get("trained_with", c.get("sampling")),
             "seed": meta.get("seed"), "iterations": meta.get("iterations"),
             "source_checkpoint": meta.get("source_checkpoint"),
             "generation_command": gen_cmd,
             "generation_commit": None,
             "generation_commit_note": "not recorded when the artifact was generated",
             "first_committed_in": first_commit(p)}
        assert "blinds=0.5/1" in game_sig, game_sig
        manifest_path(p).write_text(json.dumps(m, indent=1, sort_keys=True))
        print(f"wrote {manifest_path(p)}")


if __name__ == "__main__":
    main(sys.argv[1:])
