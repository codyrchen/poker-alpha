"""Phase 37: build the per-key confidence table for a strategy artifact.

From checkpoints of the same solver config:

* primary seed, final iteration -> visits;
* primary seed, earlier iteration -> movement (L1 of average strategies);
* other seeds, final iteration -> seed disagreement (mean pairwise L1 among
  the seeds that visited the key);
* a seeded corpus -> collision dispersion: range of the hand-strength ladder
  (0..7, /7) among corpus states mapped to the key (needs >= 5 members);
* the Phase-32 audit -> known pathological keys (premium hands folding or
  limping heavily, first-in jams > 30%).

Usage::

    python experiments/phase37_build_confidence.py --ckpt-dir DIR --config v1 \
        --final 300000 --earlier 100000 --out results/strategy/holdem_v1_seed0_confidence.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.abstraction.features import card_features  # noqa: E402
from poker_alpha.decision.solver_gate import ConfidenceTable, KeyStats  # noqa: E402
from poker_alpha.solver_config import PRIMARY_CONFIG, V2_CONFIG  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402
from poker_alpha.validation.abstraction_audit import generate_corpus  # noqa: E402


def policies(path, game):
    # Native checkpoints (pokeralpha.native_checkpoint/v1) are read directly
    # from their arrays — no extension needed; Python checkpoints go through
    # load_checkpoint as before.
    with np.load(path, allow_pickle=False) as z:
        if "format" in z.files and \
                str(z["format"][()]) == "pokeralpha.native_checkpoint/v1":
            if str(z["solver_config"][()]) != game.solver_config_signature():
                raise SystemExit(f"native checkpoint config mismatch: {path}")
            keys = z["keys"].tolist()
            off = z["action_offsets"]
            ss = z["strategy_sum"]
            out = {}
            for i, k in enumerate(keys):
                lo, hi = int(off[i]), int(off[i + 1])
                v = float(ss[lo:hi].sum())
                if v > 0:
                    out[k] = (v, ss[lo:hi] / v)
            return out
    s = load_checkpoint(path, game)
    out = {k: (float(n.strategy_sum.sum()), n.average_strategy())
           for k, n in s.infosets.items() if n.strategy_sum.sum() > 0}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--config", choices=("v1", "v2"), default="v1")
    p.add_argument("--prefix", default="locked")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--final", type=int, required=True)
    p.add_argument("--earlier", type=int, required=True,
                   help="movement baseline. Schema 1: an early checkpoint "
                        "(historical 10k). Schema 2 (--schema 2): the "
                        "previous mature milestone (recent movement)")
    p.add_argument("--schema", type=int, choices=(1, 2), default=1,
                   help="confidence schema: 2 = recent-movement semantics "
                        "(solver_confidence/v2), validated in "
                        "results/validation/confidence_signal_quality.json")
    p.add_argument("--bind-artifact", type=Path, default=None,
                   help="schema 2: record this strategy artifact's SHA-256 "
                        "so loaders reject mismatched strategy/table pairs")
    p.add_argument("--corpus-hands", type=int, default=20000)
    p.add_argument("--audit", type=Path, default=ROOT / "results" / "validation" / "preflop_audit_v1.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cfg = PRIMARY_CONFIG if a.config == "v1" else V2_CONFIG
    game = cfg.build_game()
    seeds = [int(x) for x in a.seeds.split(",")]
    ck = lambda s, it: a.ckpt_dir / f"{a.prefix}_seed{s}_it{it}.npz"  # noqa: E731
    final = {s: policies(ck(s, a.final), game) for s in seeds}
    earlier = policies(ck(seeds[0], a.earlier), game)
    print("loaded", flush=True)

    strengths = defaultdict(list)
    for st in generate_corpus(game, a.corpus_hands, 17):
        if st.street == 0:
            continue
        p0 = game.current_player(st)
        strengths[game.infoset_key(st)].append(card_features(st.holes[p0], st.board).strength)
    stats = {}
    for k, (v, pol) in final[seeds[0]].items():
        mv = float(np.abs(pol - earlier[k][1]).sum()) if k in earlier else math.nan
        pols = [pol] + [final[s][k][1] for s in seeds[1:] if k in final[s]]
        sd = float(np.mean([np.abs(x - y).sum() for i, x in enumerate(pols) for y in pols[i + 1:]])) \
            if len(pols) > 1 else math.nan
        m = strengths.get(k, [])
        col = (max(m) - min(m)) / 7.0 if len(m) >= 5 else math.nan
        stats[k] = KeyStats(v, mv, sd, col)
    patho = []
    if a.config == "v1" and a.audit.exists():
        audit = json.loads(a.audit.read_text())
        flagged = {"premium folds > 5%", "premium limps > 30%", "jams > 30% as first raise"}
        for s in audit["surprises"]:
            if flagged & set(s["flags"]):
                patho.append(audit["records"][s["state"]]["key"])
    meta = {"seeds": seeds, "final": a.final, "earlier": a.earlier,
            "corpus_hands": a.corpus_hands,
            "collision": "range of the 0..7 strength ladder among >= 5 corpus members, /7"}
    if a.schema == 2:
        meta.update({"schema": 2, "movement_mode": "recent",
                     "movement_from": a.earlier, "movement_to": a.final})
        if a.bind_artifact is not None:
            from poker_alpha.utils.provenance import file_sha256

            meta["strategy_sha256"] = file_sha256(a.bind_artifact)
            meta["strategy_artifact"] = a.bind_artifact.name
    table = ConfidenceTable(cfg.signature(), stats, tuple(sorted(set(patho))),
                            meta, schema=a.schema)
    table.save(a.out)
    arr = np.array([[s.visits, s.movement, s.seed_disagreement, s.collision] for s in stats.values()])
    print("keys", len(stats), "pathological", len(set(patho)),
          "median visits", np.nanmedian(arr[:, 0]), "median movement", np.nanmedian(arr[:, 1]),
          "median seed dis", np.nanmedian(arr[:, 2]), "collision known", int(np.sum(~np.isnan(arr[:, 3]))))


if __name__ == "__main__":
    main()
