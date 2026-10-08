"""One normalized analysis path for every input source (Phase 31).

    manual dict / simulation / hand history / screenshot
        -> ObservedTableState            (adapters; nothing is invented)
        -> validate                      (rules-level consistency)
        -> recommend_action              (solver -> rollout -> heuristic)
        -> DecisionReport                (provenance, source cascade,
                                          uncertainty by source)

Every ``observe_*`` function returns an :class:`Observation` (state plus how
it was obtained); :func:`analyze` is the only entry into the decision engine.
The screenshot path is read-only: it reads pixels from an image it is given
and never clicks, types, controls a browser or submits actions. Real-time
use belongs only in private, play-money or test environments where such
assistance is permitted; otherwise use post-hand analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

from .decision import DecisionConfig, DecisionReport, recommend_action
from .decision.strategy import LookupMiss, SolverStrategyProvider
from .holdem import validate
from .holdem.adapters import ManualStateAdapter, SimulationStateAdapter
from .holdem.observed import ObservedTableState

SOURCES = ("manual", "simulation", "hand_history", "screenshot")


@dataclass(frozen=True)
class Observation:
    state: ObservedTableState
    source: str                                   # one of SOURCES
    observer_confidence: Optional[float] = None   # screenshots only
    notes: tuple = field(default_factory=tuple)


def observe_manual(data: Mapping[str, Any]) -> Observation:
    st = ManualStateAdapter.from_dict(dict(data))
    return Observation(replace(st, source="manual"), "manual")


def observe_simulation(table_state, hero_seat: int, chip_unit: float = 1.0) -> Observation:
    st = SimulationStateAdapter.observe(table_state, hero_seat, chip_unit=chip_unit)
    return Observation(replace(st, source="simulation"), "simulation")


def observe_hand_history(events: Sequence, decision: int = 0,
                         hero_seat: Optional[int] = None,
                         chip_scale: int = 100) -> Observation:
    """The hero's ``decision``-th decision point of a parsed hand
    (:func:`poker_alpha.history.load_hands`)."""
    from .history import replay_hand

    res = replay_hand(events, hero_seat=hero_seat, chip_scale=chip_scale)
    if not res.decisions:
        raise ValueError("hand history contains no hero decision")
    d = res.decisions[decision]
    note = f"hero actually chose {d.action}" + (f" to {d.amount:g}" if d.amount else "")
    return Observation(replace(d.state, source="hand_history"), "hand_history",
                       notes=(note,) + tuple(res.warnings))


def observe_screenshot(image: Any, calibration=None, small_blind: float = 0.5,
                       big_blind: float = 1.0, frames: int = 3,
                       seats: int = 6) -> Observation:
    """Recognize a table from an image (PIL image or path). Fusion needs
    agreement across frames, so the same image is fed ``frames`` times."""
    from .observer.fusion import StateTracker
    from .observer.pokernow import PokerNowStyleAdapter, default_layout

    if isinstance(image, (str, Path)):
        from PIL import Image

        image = Image.open(image).convert("RGB")
    cal = calibration or default_layout(seats)
    adapter = PokerNowStyleAdapter(cal)
    tracker = StateTracker(cal, small_blind, big_blind)
    for _ in range(max(frames, 1)):
        tracker.update(adapter.read_frame(image))
    st = tracker.to_observed_state()
    return Observation(replace(st, source="screenshot"), "screenshot",
                       observer_confidence=tracker.critical_confidence(),
                       notes=tuple(tracker.tracked().warnings)
                       + ("real PokerNow recognition accuracy is not validated",))


def load_solver(path: Optional[Union[str, Path]], config=None,
                min_visits: float = 20.0):
    """Provider or :class:`LookupMiss` (``config_mismatch`` /
    ``incompatible_checkpoint``) for a strategy artifact or checkpoint."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return LookupMiss(f"strategy file {path} not found", "incompatible_checkpoint")
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            is_artifact = "format" in z.files
    except (OSError, ValueError) as exc:
        return LookupMiss(f"cannot read {path}: {exc}", "incompatible_checkpoint")
    if is_artifact:
        return SolverStrategyProvider.from_artifact(path, config, min_visits)
    return SolverStrategyProvider.from_checkpoint(path, config, min_visits)


def analyze(obs: Observation, config: Optional[DecisionConfig] = None,
            solver=None, **kwargs) -> DecisionReport:
    """The single entry into the decision engine for every source.

    ``solver`` is a provider, a :class:`LookupMiss` from :func:`load_solver`,
    or ``None``; it overrides ``config.solver``.
    """
    cfg = config or DecisionConfig()
    if solver is not None:
        if isinstance(solver, LookupMiss):
            cfg = replace(cfg, solver=None, solver_unavailable=solver)
        else:
            cfg = replace(cfg, solver=solver, solver_unavailable=None)
    if obs.observer_confidence is not None:
        cfg = replace(cfg, observer_confidence=obs.observer_confidence)
    report = recommend_action(obs.state, config=cfg, **kwargs)
    details = dict(report.details)
    details["input_source"] = obs.source
    if obs.notes:
        details["input_notes"] = list(obs.notes)
    return replace(report, details=details)


def validation_issues(obs: Observation) -> List[str]:
    return [f"{i.severity}: {i.message}" for i in validate(obs.state)]


__all__ = ["Observation", "SOURCES", "observe_manual", "observe_simulation",
           "observe_hand_history", "observe_screenshot", "load_solver", "analyze",
           "validation_issues"]
