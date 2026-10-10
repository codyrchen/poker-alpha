"""Solver-use gate (Phase 37): accept, down-weight or reject a solver lookup.

An abstract key existing in a strategy file is not enough. For every
information set, a *confidence table* built offline from the training runs
(`experiments/phase37_build_confidence.py`) stores

* ``visits``            non-updating-player visits in the primary run,
* ``movement``          L1 between the primary run's average strategy at an
                        earlier and the final checkpoint,
* ``seed_disagreement`` mean pairwise L1 between independently seeded runs,
* ``collision``         dispersion of the hand-strength ladder among the
                        corpus states merged into the key (0..1),

and :func:`gate` turns them into ``SOLVER_ACCEPT``, ``SOLVER_LOW_CONFIDENCE``
or ``SOLVER_REJECT`` with machine-readable reasons. Thresholds for visits,
movement and seed disagreement come from measured true strategy error in
exact games (``results/validation/solver_gate_calibration.json``); the
collision threshold is a documented heuristic (no exact reference exists
for the full abstraction).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ACCEPT, LOW, REJECT = "SOLVER_ACCEPT", "SOLVER_LOW_CONFIDENCE", "SOLVER_REJECT"

REASONS = {
    "LOW_VISIT_COUNT": "information set visited too rarely",
    "HIGH_SEED_DISAGREEMENT": "independently seeded runs disagree",
    "UNSTABLE_ACROSS_CHECKPOINTS": "policy still moving between checkpoints",
    "HIGH_COLLISION_DISPERSION": "the abstract key merges strategically different hands",
    "CONFIG_MISMATCH": "strategy trained under a different solver config",
    "INCOMPATIBLE_CHECKPOINT": "strategy file unreadable or unsupported",
    "UNSEEN_STATE": "information set never visited in training",
    "OUTSIDE_ABSTRACTION": "spot cannot be mapped into the abstract game",
    "KNOWN_PATHOLOGICAL_BUCKET": "key flagged by the strategy audit",
    "NO_STABILITY_DATA": "no checkpoint / seed stability data for this key",
    # gate v2 (Phase 64): spot-level reasons added at lookup time; they can
    # only lower the status (ACCEPT -> LOW), never raise it.
    "OFF_TREE_TRANSLATION": "observed bet sizes were translated onto the abstract tree",
    "ILLEGAL_SIZE_MASS": "a large share of the solver's mass was on sizes below the "
                         "NLHE minimum and was removed",
    # Phase 58: exact flop / turn subgames show the compact encoders (v1 and
    # the v2 release) merging strategically different hands (lifted
    # strategies 1.4-9.6 BB/hand exploitable vs ~0.02 for raw keys).
    "STREET_ABSTRACTION_ERROR": "flop / turn keys of this abstraction measurably merge "
                                "strategically different hands (exact subgame study)",
}

#: streets (0 preflop .. 3 river) whose lookups are capped at LOW confidence
#: for encoders measured to merge strategically different hands there.
LOW_CONFIDENCE_STREETS = {"CompactHoldemEncoder": (1, 2)}   # v1 and v2 releases

#: removed illegal-size mass at or above this lowers the status (heuristic).
ILLEGAL_MASS_LOW = 0.2


def downgrade(decision: dict, reason: str) -> dict:
    """Add a LOW reason to a gate decision dict; ACCEPT becomes LOW, REJECT stays."""
    d = dict(decision, reasons=list(decision.get("reasons", [])))
    if reason not in d["reasons"]:
        d["reasons"].append(reason)
    if d.get("status") == ACCEPT:
        d["status"] = LOW
    return d

CALIBRATION = Path(__file__).resolve().parents[2] / "results" / "validation" / "solver_gate_calibration.json"


@dataclass(frozen=True)
class GateThresholds:
    reject_visits_below: float = 20.0
    low_visits_below: float = 100.0
    reject_seed_disagreement: float = 1.0
    low_seed_disagreement: float = 0.5
    reject_movement: float = 0.8
    low_movement: float = 0.3
    low_collision: float = 3 / 7          # >= 3 strength rungs mixed (heuristic)
    source: str = "defaults"

    @classmethod
    def calibrated(cls, path: Path = CALIBRATION) -> "GateThresholds":
        """Thresholds from the exact-game calibration where it found one,
        defaults otherwise (recorded in ``source``)."""
        try:
            t = json.loads(Path(path).read_text())["thresholds"]
        except (OSError, ValueError, KeyError):
            return cls()
        d = cls()

        def pick(key, default):
            v = t.get(key)
            return float(v) if v is not None else default
        # Visits did not predict true error in the exact games (calibration
        # file), so the visit rule is a documented sanity floor, not a
        # calibrated threshold: reject below 20 visits, low-confidence below
        # max(calibrated, 20).
        rv = pick("reject_visits_below", 20.0)
        lv = max(pick("low_conf_visits_below", rv), rv)
        return cls(reject_visits_below=rv, low_visits_below=lv,
                   reject_seed_disagreement=pick("reject_seed_disagreement_at_or_above",
                                                 d.reject_seed_disagreement),
                   low_seed_disagreement=pick("low_conf_seed_disagreement_at_or_above",
                                              d.low_seed_disagreement),
                   reject_movement=pick("reject_movement_at_or_above", d.reject_movement),
                   low_movement=pick("low_conf_movement_at_or_above", d.low_movement),
                   source=str(path.name))


@dataclass(frozen=True)
class KeyStats:
    visits: float
    movement: float = math.nan
    seed_disagreement: float = math.nan
    collision: float = math.nan


@dataclass(frozen=True)
class GateDecision:
    status: str
    reasons: Tuple[str, ...]
    signals: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "reasons": list(self.reasons), "signals": dict(self.signals)}


def _f(x) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def gate(stats: Optional[KeyStats], th: GateThresholds,
         pathological: bool = False) -> GateDecision:
    if stats is None or stats.visits <= 0:
        return GateDecision(REJECT, ("UNSEEN_STATE",), {"visits": 0.0})
    sig = {"visits": _f(stats.visits), "movement": _f(stats.movement),
           "seed_disagreement": _f(stats.seed_disagreement), "collision": _f(stats.collision)}
    reject, low = [], []
    if stats.visits < th.reject_visits_below:
        reject.append("LOW_VISIT_COUNT")
    elif stats.visits < th.low_visits_below:
        low.append("LOW_VISIT_COUNT")
    sd, mv, col = stats.seed_disagreement, stats.movement, stats.collision
    if math.isnan(sd) and math.isnan(mv):
        low.append("NO_STABILITY_DATA")
    if not math.isnan(sd):
        if sd >= th.reject_seed_disagreement:
            reject.append("HIGH_SEED_DISAGREEMENT")
        elif sd >= th.low_seed_disagreement:
            low.append("HIGH_SEED_DISAGREEMENT")
    if not math.isnan(mv):
        if mv >= th.reject_movement:
            reject.append("UNSTABLE_ACROSS_CHECKPOINTS")
        elif mv >= th.low_movement:
            low.append("UNSTABLE_ACROSS_CHECKPOINTS")
    if not math.isnan(col) and col >= th.low_collision:
        low.append("HIGH_COLLISION_DISPERSION")
    if pathological:
        low.append("KNOWN_PATHOLOGICAL_BUCKET")
    if reject:
        return GateDecision(REJECT, tuple(reject + low), sig)
    if low:
        return GateDecision(LOW, tuple(low), sig)
    return GateDecision(ACCEPT, (), sig)


class ConfidenceTable:
    """Per-key stability statistics.

    Schema v1 (``pokeralpha.solver_confidence/v1``): ``movement`` is the L1
    between a very early checkpoint (10k) and the final one — measured in the
    final-trust project to penalize mature strategies (it tracks distance
    traveled, not instability) and to carry no EV-regret signal at any
    horizon.

    Schema v2 (``pokeralpha.solver_confidence/v2``): identical layout and
    thresholds, but ``movement`` is the RECENT movement — L1 between the
    final checkpoint and the previous mature milestone (recorded in
    ``meta["movement_from"/"movement_to"]``). Held-out exact-game
    calibration: the unchanged ACCEPT thresholds then give ~1.75x the
    coverage at equal accepted EV regret and no worse wrong-action rate
    (results/validation/confidence_signal_quality.json). v2 tables may also
    bind to their strategy artifact via ``meta["strategy_sha256"]``, which
    loaders must verify (no silent strategy/table mismatches).
    """

    FORMAT = "pokeralpha.solver_confidence/v1"
    FORMAT_V2 = "pokeralpha.solver_confidence/v2"

    def __init__(self, config_signature: str, stats: Dict[str, KeyStats],
                 pathological: Tuple[str, ...] = (), meta: Optional[dict] = None,
                 schema: int = 1) -> None:
        self.config_signature = config_signature
        self.stats = stats
        self.pathological = frozenset(pathological)
        self.meta = meta or {}
        if schema not in (1, 2):
            raise ValueError(f"unknown confidence schema {schema}")
        self.schema = schema

    def get(self, key: str) -> Optional[KeyStats]:
        return self.stats.get(key)

    def save(self, path) -> Path:
        keys = sorted(self.stats)
        arr = np.array([[self.stats[k].visits, self.stats[k].movement, self.stats[k].seed_disagreement,
                         self.stats[k].collision] for k in keys], dtype=np.float32).reshape(-1, 4)
        path = Path(path)
        fmt = self.FORMAT_V2 if self.schema == 2 else self.FORMAT
        np.savez_compressed(path, format=np.array(fmt), config_sig=np.array(self.config_signature),
                            meta=np.array(json.dumps(self.meta, sort_keys=True)),
                            keys=np.array(keys, dtype=np.str_), stats=arr,
                            pathological=np.array(sorted(self.pathological), dtype=np.str_))
        return path

    @classmethod
    def load(cls, path, config_signature: Optional[str] = None) -> "ConfidenceTable":
        with np.load(Path(path), allow_pickle=False) as z:
            fmt = str(z["format"][()])
            if fmt == cls.FORMAT:
                schema = 1
            elif fmt == cls.FORMAT_V2:
                schema = 2
            else:
                raise ValueError("not a solver confidence table")
            sig = str(z["config_sig"][()])
            if config_signature is not None and sig != config_signature:
                raise ValueError(f"confidence table for {sig}, strategy is {config_signature}")
            keys = z["keys"].tolist()
            rows = z["stats"].astype(float).tolist()
            stats = {k: KeyStats(*r) for k, r in zip(keys, rows)}
            return cls(sig, stats, tuple(str(k) for k in z["pathological"]),
                       json.loads(str(z["meta"][()])), schema=schema)
