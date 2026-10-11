"""Decision Receipts: a reproducibility record for every analysis.

A receipt is a normalized, versioned summary of everything that produced a
recommendation: the state fingerprint, the strategy artifact SHA, the
solver/config signatures, the engine commit, the method and gate verdict,
action-mapping quality, and the sampling seeds. Its short hash
(``PA-XXXXXXXX``) is a **reproducibility ID** — a stable content address
for "this input, this artifact, this configuration", not a cryptographic
proof of anything.

``verify_receipt`` re-runs the analysis from the receipt's recorded inputs
and checks that the recommendation, mix and gate verdict match — possible
because every stochastic component records its seed.

Privacy: receipts contain no filesystem paths, usernames, screenshots or
hand-history identifiers — only content hashes, signatures and the state
itself (which the user supplied).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ..decision import DecisionConfig, DecisionReport

RECEIPT_FORMAT = "pokeralpha.receipt/v1"

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _artifact_sha(rel_path: str) -> Optional[str]:
    p = _ROOT / rel_path
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _engine_commit() -> Optional[str]:
    from ..observer.session import git_commit

    return git_commit()


def state_fingerprint(state_dict: Dict[str, Any]) -> str:
    blob = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_receipt(report: DecisionReport, state_dict: Dict[str, Any],
                  config: Optional[DecisionConfig] = None,
                  mapping_quality: Optional[str] = None) -> Dict[str, Any]:
    """The normalized receipt for one analysis."""
    from ..solver_config import RELEASE_STRATEGY
    from .alignment import solver_state

    cfg = config or DecisionConfig()
    details = report.details or {}
    info = details.get("solver", {})
    used_solver = bool(info.get("used")) or info.get("confidence") not in (
        "not configured",)
    receipt: Dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "state_fingerprint": state_fingerprint(state_dict),
        "state": state_dict,
        "engine_commit": _engine_commit(),
        "strategy_artifact_sha256": _artifact_sha(RELEASE_STRATEGY)
        if used_solver else None,
        "config": {
            "equity_simulations": cfg.equity_simulations,
            "rollout_simulations": cfg.rollout_simulations,
            "seed": cfg.seed,
            "abstraction_signature": cfg.abstraction.signature(),
            "rollout_response": cfg.rollout_response,
            "rollout_depth": cfg.rollout_depth,
            "default_model": cfg.default_model,
        },
        "method": report.method,
        "solver_gate": {
            "state": solver_state(report),
            "reasons": list(info.get("reasons", ())),
        },
        "infoset_key": details.get("infoset"),
        "training_visits": details.get("visits"),
        "mapping_quality": mapping_quality,
        "result": {
            "recommended": report.recommended,
            "recommended_mix": {k: round(float(v), 10)
                                for k, v in (report.recommended_mix or {}).items()},
            "confidence": report.confidence,
        },
    }
    receipt["receipt_id"] = receipt_id(receipt)
    return receipt


def receipt_id(receipt: Dict[str, Any]) -> str:
    """Stable short reproducibility ID over the normalized content
    (excluding the id field itself)."""
    body = {k: v for k, v in receipt.items() if k != "receipt_id"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return "PA-" + hashlib.sha256(blob.encode()).hexdigest()[:8].upper()


def verify_receipt(receipt: Dict[str, Any], solver=None) -> Dict[str, Any]:
    """Re-run the analysis from the receipt's recorded inputs and compare.

    Returns {"reproduced": bool, "mismatches": [...]} — exact comparison is
    possible because seeds and configuration are part of the receipt. A
    different strategy artifact (SHA mismatch) is reported rather than
    silently producing different numbers.
    """
    from ..pipeline import analyze, load_solver, observe_manual
    from ..solver_config import RELEASE_STRATEGY
    from .alignment import solver_state

    mismatches = []
    want_sha = receipt.get("strategy_artifact_sha256")
    if want_sha:
        have = _artifact_sha(RELEASE_STRATEGY)
        if have != want_sha:
            return {"reproduced": False,
                    "mismatches": [f"strategy artifact SHA differs "
                                   f"(receipt {want_sha[:12]}…, local "
                                   f"{str(have)[:12]}…)"]}
    c = receipt["config"]
    cfg = DecisionConfig(equity_simulations=c["equity_simulations"],
                         rollout_simulations=c["rollout_simulations"],
                         seed=c["seed"],
                         rollout_response=c.get("rollout_response", "behavior"),
                         rollout_depth=c.get("rollout_depth", "street"),
                         default_model=c.get("default_model", "regular"))
    rep = analyze(observe_manual(receipt["state"]), config=cfg,
                  solver=solver if solver is not None
                  else load_solver(str(_ROOT / RELEASE_STRATEGY)))
    want = receipt["result"]
    if rep.recommended != want["recommended"]:
        mismatches.append(f"recommended: {rep.recommended!r} != "
                          f"{want['recommended']!r}")
    got_mix = {k: round(float(v), 10)
               for k, v in (rep.recommended_mix or {}).items()}
    if got_mix != want["recommended_mix"]:
        mismatches.append("recommended_mix differs")
    got_gate = solver_state(rep)
    if got_gate != receipt["solver_gate"]["state"]:
        mismatches.append(f"gate: {got_gate} != "
                          f"{receipt['solver_gate']['state']}")
    return {"reproduced": not mismatches, "mismatches": mismatches}


__all__ = ["RECEIPT_FORMAT", "build_receipt", "receipt_id",
           "state_fingerprint", "verify_receipt"]
