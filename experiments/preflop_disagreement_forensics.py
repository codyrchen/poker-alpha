"""Phase 11 (final-trust): preflop disagreement forensics.

For every one of the 169 canonical classes in the key situations, across the
native lineage milestones 200k / 300k / 1M (comparable seeds 0-2):

* per-seed policy, pairwise L1 disagreement, visits (seed 0);
* recent movement (consecutive checkpoints of seed 0);
* at 1M, measured EV data from the policy-vs-value studies (btn169 +
  canonical runs): action margin, cross-seed regret, best-action agreement;
* a diagnosis per state:

  - UNDERVISITED        visits at 1M below the gate floor (20)
  - STABLE              L1 < 0.3 at 1M
  - CONSEQUENTIAL       high L1 AND cross-seed regret above
                        max(0.5% pot, 3 * max q SE) (EV study states only)
  - NEAR_EQUIVALENT     high L1, EV consequence measured below threshold
  - SAMPLING_LIMITED    high L1 but still declining strongly with training
                        (L1@1M < 0.6 * L1@200k) — more iterations help
  - PERSISTENT_MIXING   high L1, not declining, no EV measurement available

Writes results/validation/preflop_disagreement_forensics.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from native_300k_study import CANONICAL_SITUATIONS, canonical_state  # noqa: E402
from phase37_build_confidence import policies  # noqa: E402
from poker_alpha.abstraction.cards import all_preflop_classes  # noqa: E402
from poker_alpha.solver_config import V2_CONFIG  # noqa: E402

CK = ROOT / "results/native_training"
MILESTONES = [200_000, 300_000, 1_000_000]
RECENT = {200_000: 150_000, 300_000: 250_000, 1_000_000: 750_000}
SEEDS = (0, 1, 2)
OUT = ROOT / "results/validation/preflop_disagreement_forensics.json"


def ck(seed, it):
    return CK / f"seed{seed}" / f"v2_native_seed{seed}_it{it}.npz"


def main() -> None:
    game = V2_CONFIG.build_game()
    pol = {}
    for s in SEEDS:
        for m in MILESTONES:
            pol[(s, m)] = policies(ck(s, m), game)
    pol_recent = {m: policies(ck(0, RECENT[m]), game) for m in MILESTONES}
    print("checkpoints loaded", flush=True)

    ev_lookup = {}
    for f in ("results/validation/policy_vs_value_disagreement_native_1m.json",
              "results/validation/preflop169_value_native_1m.json"):
        p = ROOT / f
        if p.exists():
            d = json.loads(p.read_text())
            for row in d["states"]:
                ses = [max(e["q_se"].values()) for e in row["envs"].values()
                       if e.get("q_se")]
                ev_lookup[row["id"]] = {
                    "pot": row["pot"],
                    "action_margin_mean": row.get("action_margin_mean"),
                    "cross_regret_max": row.get("cross_regret_max"),
                    "best_action_agreement": row.get("best_action_agreement"),
                    "max_q_se": max(ses) if ses else None}

    states = []
    for sit, tokens, actor in CANONICAL_SITUATIONS:
        for cls in all_preflop_classes():
            st = canonical_state(game, cls, tokens, actor)
            key = game.infoset_key(st)
            legal = game.legal_actions(st)
            row = {"id": f"{sit}::{cls}", "situation": sit, "class": cls,
                   "key": key}
            per_m = {}
            for m in MILESTONES:
                pols = [pol[(s, m)].get(key) for s in SEEDS]
                arrs = [p[1] for p in pols if p is not None]
                if len(arrs) < 2 or any(len(a) != len(legal) for a in arrs):
                    per_m[str(m)] = None
                    continue
                l1s = [float(np.abs(a - b).sum()) for i, a in enumerate(arrs)
                       for b in arrs[i + 1:]]
                hit0 = pol[(0, m)].get(key)
                rec = pol_recent[m].get(key)
                per_m[str(m)] = {
                    "seed_l1_mean": round(float(np.mean(l1s)), 4),
                    "visits_seed0": round(hit0[0], 1) if hit0 else 0.0,
                    "recent_movement": (round(float(np.abs(hit0[1] - rec[1]).sum()), 4)
                                        if hit0 and rec is not None
                                        and len(rec[1]) == len(hit0[1]) else None),
                }
                if m == 1_000_000 and hit0 is not None:
                    per_m[str(m)]["policy_seed0"] = {
                        a: round(float(x), 4) for a, x in zip(legal, hit0[1])}
            row["milestones"] = per_m
            row["ev"] = ev_lookup.get(row["id"])

            # diagnosis
            m1 = per_m.get("1000000")
            m200 = per_m.get("200000")
            if m1 is None:
                diag = "NO_DATA"
            elif m1["visits_seed0"] < 20:
                diag = "UNDERVISITED"
            elif m1["seed_l1_mean"] < 0.3:
                diag = "STABLE"
            else:
                ev = row["ev"]
                if ev and ev["cross_regret_max"] is not None and ev["max_q_se"]:
                    thresh = max(0.005 * ev["pot"], 3 * ev["max_q_se"], 0.05)
                    diag = ("CONSEQUENTIAL" if ev["cross_regret_max"] >= thresh
                            else "NEAR_EQUIVALENT")
                elif (m200 and m200["seed_l1_mean"] > 0
                      and m1["seed_l1_mean"] < 0.6 * m200["seed_l1_mean"]):
                    diag = "SAMPLING_LIMITED"
                else:
                    diag = "PERSISTENT_MIXING"
            row["diagnosis"] = diag
            states.append(row)

    counts = Counter(r["diagnosis"] for r in states)
    by_sit = {}
    for sit, _, _ in CANONICAL_SITUATIONS:
        sub = [r for r in states if r["situation"] == sit]
        by_sit[sit] = dict(Counter(r["diagnosis"] for r in sub))
    doc = {"format": "pokeralpha.preflop_disagreement_forensics/v1",
           "milestones": MILESTONES, "seeds": list(SEEDS),
           "lineage": "native (xoshiro seeds 0-2)",
           "diagnosis_counts": dict(counts),
           "by_situation": by_sit,
           "states": states}
    OUT.write_text(json.dumps(doc, indent=1))
    print("diagnoses:", dict(counts))
    for sit, c in by_sit.items():
        print(f"  {sit}: {c}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
