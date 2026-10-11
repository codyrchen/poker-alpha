"""Study/session analysis: review, strict regret semantics, alignment,
summaries, pattern detection and local exports.

The analysis suite never invents evaluation of its own: every decision is
re-analysed by the real engine (``poker_alpha.pipeline.analyze``), and every
aggregate keeps methodology boundaries — solver frequencies are never read
as EVs, and EV regret exists only where :func:`regret_eligibility` says the
comparison is valid.
"""

from .alignment import (CATEGORY_TEXT, StrategyAlignment, solver_state,
                        strategy_alignment)
from .export import (EXPORT_FORMAT, record_to_dict, review_to_csv,
                     review_to_json, review_to_markdown)
from .patterns import Pattern, detect_patterns
from .regret import INELIGIBLE_REASONS, RegretEligibility, regret_eligibility
from .review import (CATEGORIES, DecisionRecord, SessionReview, categorize,
                     review_decision, review_hands)
from .summary import (GroupStats, SessionSummary, confidence_report,
                      summarize)

__all__ = [
    "CATEGORY_TEXT", "StrategyAlignment", "solver_state", "strategy_alignment",
    "EXPORT_FORMAT", "record_to_dict", "review_to_csv", "review_to_json",
    "review_to_markdown", "Pattern", "detect_patterns",
    "INELIGIBLE_REASONS", "RegretEligibility", "regret_eligibility",
    "CATEGORIES", "DecisionRecord", "SessionReview", "categorize",
    "review_decision", "review_hands",
    "GroupStats", "SessionSummary", "confidence_report", "summarize",
]
