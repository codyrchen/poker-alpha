"""Decision support: candidate actions, EVs, provenance and uncertainty."""

from .recommend import (DecisionConfig, betting_context,
                        infer_opponent_range, recommend_action)
from .report import SOURCES, CandidateAction, DecisionReport, RangeSummary
from .strategy import LookupMiss, SolverLookup, SolverStrategyProvider

__all__ = [
    "DecisionConfig", "betting_context", "infer_opponent_range",
    "recommend_action", "SOURCES", "CandidateAction", "DecisionReport",
    "RangeSummary", "LookupMiss", "SolverLookup", "SolverStrategyProvider",
]
