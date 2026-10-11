"""Opponent-model sensitivity: is the recommendation robust to how the
opponent is modelled?

The same state is analysed under each EXISTING behavior archetype
(``poker_alpha.opponent.behavior.ARCHETYPE_MODELS``: regular, nit,
calling_station, maniac). No archetype is presented as correct — the
product question is whether the advice SURVIVES not knowing which one is.

Classification rule (documented, pinned by tests). Let ``top_m`` be the
highest-probability action under model ``m`` and ``TV(m, base)`` the total
variation distance between the action distributions:

    ROBUST            every model agrees on the top action and
                      max TV < 0.20
    SENSITIVE         every model agrees on the top action but
                      max TV >= 0.20, OR exactly one model disagrees
    HIGHLY SENSITIVE  two or more models disagree on the top action

Solver-method results are compared by their mixes; rollout results by
selection probabilities. EVs are never compared across different decision
methods — when the methods themselves differ between models, that is
reported as its own signal (``methods_differ``) and the verdict is at
least SENSITIVE.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from ..decision import DecisionConfig, DecisionReport
from ..opponent.behavior import ARCHETYPE_MODELS
from ..pipeline import analyze, observe_manual
from .alignment import solver_state

ROBUST_TV_BELOW = 0.20

VERDICTS = ("ROBUST", "SENSITIVE", "HIGHLY SENSITIVE")


@dataclass(frozen=True)
class ModelOutcome:
    model: str
    top_action: Optional[str]
    distribution: Dict[str, float]      # label -> probability (mix or selection)
    method: str
    source_kind: str
    solver_state: str
    confidence: str
    evs: Dict[str, Optional[float]]     # label -> rollout EV (same-model only)


@dataclass(frozen=True)
class SensitivityResult:
    outcomes: Tuple[ModelOutcome, ...]
    verdict: str
    max_tv: float
    top_actions: Dict[str, Optional[str]]
    methods_differ: bool
    note: str


def _distribution(report: DecisionReport) -> Dict[str, float]:
    mix = dict(report.recommended_mix or {})
    if mix:
        return mix
    return {c.label: c.probability for c in report.candidates
            if c.probability}


def _tv(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def analyze_under_models(state: dict, config: Optional[DecisionConfig] = None,
                         solver=None,
                         models: Optional[List[str]] = None) -> SensitivityResult:
    names = models or list(ARCHETYPE_MODELS)
    outcomes: List[ModelOutcome] = []
    for name in names:
        cfg = replace(config or DecisionConfig(), default_model=name)
        rep = analyze(observe_manual(state), config=cfg, solver=solver)
        dist = _distribution(rep)
        top = max(dist, key=dist.get) if dist else rep.recommended
        m = rep.method.lower()
        kind = ("solver" if ("solver" in m or "abstraction" in m) else
                "rollout" if "rollout" in m else "heuristic")
        outcomes.append(ModelOutcome(
            model=name, top_action=top, distribution=dist,
            method=rep.method, source_kind=kind,
            solver_state=solver_state(rep), confidence=rep.confidence,
            evs={c.label: c.ev_bb for c in rep.candidates}))
    tops = {o.model: o.top_action for o in outcomes}
    base = outcomes[0]
    max_tv = max((_tv(base.distribution, o.distribution)
                  for o in outcomes[1:]), default=0.0)
    methods_differ = len({o.source_kind for o in outcomes}) > 1
    distinct = {t for t in tops.values()}
    disagreeing = sum(1 for o in outcomes[1:]
                      if o.top_action != base.top_action)
    if len(distinct) == 1 and max_tv < ROBUST_TV_BELOW and not methods_differ:
        verdict = "ROBUST"
    elif disagreeing >= 2:
        verdict = "HIGHLY SENSITIVE"
    else:
        verdict = "SENSITIVE"
    note = ("No opponent model here is claimed correct; the verdict only "
            "says whether the advice survives not knowing which model is.")
    if methods_differ:
        note += (" The decision METHOD itself changes across models, so "
                 "outputs are compared by distribution and status only — "
                 "never by EV across methods.")
    return SensitivityResult(tuple(outcomes), verdict, max_tv, tops,
                             methods_differ, note)


__all__ = ["SensitivityResult", "ModelOutcome", "analyze_under_models",
           "ROBUST_TV_BELOW", "VERDICTS"]
