"""Benchmark metrics for a parsed policy over the Kuhn infosets.

All exact quantities use the project's existing exact evaluation:
exploitability via best response on the real game tree; the comparison
strategy is deterministic vanilla CFR (the same family PokerAlpha uses).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ..games.kuhn import KuhnPoker
from ..solvers.cfr import CFRSolver
from ..solvers.evaluation import exploitability
from .dataset import INFOSETS


def _reference_strategy(iterations: int = 20000):
    solver = CFRSolver(KuhnPoker())
    solver.train(iterations)
    return solver.average_strategy()


def evaluate_policy(policy: Dict[str, Dict[str, float]],
                    reference: Optional[Dict] = None) -> dict:
    """Exact metrics for a complete Kuhn policy {infoset: {action: p}}.

    Missing infosets are filled uniformly and counted — an agent that
    answered only some states is not silently graded as if complete.
    """
    missing = [k for k in INFOSETS if k not in policy]
    full = {k: policy.get(k, {"p": 0.5, "b": 0.5}) for k in INFOSETS}
    game = KuhnPoker()
    expl = exploitability(game, full)
    ref = reference or _reference_strategy()
    tv = sum(0.5 * sum(abs(full[k].get(a, 0.0) - ref[k].get(a, 0.0))
                       for a in ("p", "b")) for k in INFOSETS) / len(INFOSETS)
    ent = 0.0
    deterministic = 0
    for k in INFOSETS:
        probs = [full[k].get(a, 0.0) for a in ("p", "b")]
        ent += -sum(p * math.log2(p) for p in probs if p > 0)
        deterministic += max(probs) >= 0.999
    return {
        "exploitability_chips_per_hand": round(expl, 6),
        "mean_tv_distance_to_cfr_reference": round(tv, 6),
        "mean_entropy_bits": round(ent / len(INFOSETS), 4),
        "deterministic_infosets": deterministic,
        "total_infosets": len(INFOSETS),
        "missing_infosets": missing,
        "reference": "vanilla CFR, 20,000 iterations (deterministic)",
    }


def paraphrase_instability(by_template: Dict[str, Dict[str, Dict[str, float]]]) -> dict:
    """Mean TV distance between the policies induced by different prompt
    templates for the SAME infosets — semantic equivalence should give 0."""
    names = sorted(by_template)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(by_template[a]) & set(by_template[b])
            if not common:
                continue
            tv = sum(0.5 * sum(abs(by_template[a][k].get(x, 0.0)
                                   - by_template[b][k].get(x, 0.0))
                               for x in ("p", "b")) for k in common) / len(common)
            pairs.append({"templates": [a, b], "infosets": len(common),
                          "mean_tv": round(tv, 6)})
    return {"pairs": pairs,
            "note": "0 = identical behavior under paraphrase; larger = "
                    "prompt-sensitive"}


__all__ = ["evaluate_policy", "paraphrase_instability"]
