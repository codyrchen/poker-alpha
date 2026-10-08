"""Locked heads-up Hold'em solver configuration (Phase 27).

``HoldemSolverConfig`` pins every choice that shapes a trained strategy:
information-state encoder, card-feature tables, bet menu, stack, blinds,
raise cap, MCCFR sampling variant and the (offline-only) reference range.
Its :meth:`~HoldemSolverConfig.signature` is written into every checkpoint
trained from it (see :mod:`poker_alpha.solvers.serialize`) and checked on
load, so a strategy can never be read back under a different abstraction.

PRIMARY SOLVER ENCODER: ``CompactHoldemEncoder`` (abstract betting context).
**IMPERFECT RECALL — NO STANDARD CFR EQUILIBRIUM GUARANTEE.** It was chosen
in Phase 26/27 because it is the only measured candidate that gets canonical
states revisited (``docs/validation.md`` section 8,
``results/validation/solver_abstraction_selection.json``). Strategies trained
on it are abstract heuristic strategies, not equilibria of any game.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

CONFIG_VERSION = 1

PRIMARY_SOLVER_ENCODER = "compact"
PRIMARY_SOLVER_ENCODER_RECALL = "IMPERFECT RECALL — NO STANDARD CFR EQUILIBRIUM GUARANTEE"

SAMPLING_VARIANTS = ("external-sampling-mccfr",)

_ENCODERS = ("bucket", "transition", "transition_abstract", "compact", "compact_exact")


def make_encoder(name: str):
    """Encoder instance for a registered name."""
    from .abstraction.holdem import HoldemBucketEncoder
    from .abstraction.holdem_v2 import CompactHoldemEncoder, TransitionHoldemEncoder

    factories = {
        "bucket": HoldemBucketEncoder,
        "transition": lambda: TransitionHoldemEncoder("exact"),
        "transition_abstract": lambda: TransitionHoldemEncoder("abstract"),
        "compact": lambda: CompactHoldemEncoder("abstract"),
        "compact_exact": lambda: CompactHoldemEncoder("exact"),
    }
    if name not in factories:
        raise ValueError(f"unknown encoder {name!r}; known: {sorted(factories)}")
    return factories[name]()


def card_feature_tables_digest() -> str:
    """Hash of the card-feature lookup tables, so a ladder edit without a
    version bump still changes the config signature."""
    from .abstraction import betting_history, features

    blob = json.dumps({
        "cards": features.CARD_FEATURES_VERSION,
        "transitions": features.TRANSITION_VERSION,
        "betting": betting_history.BETTING_HISTORY_VERSION,
        "strength": features.MADE_STRENGTH,
        "draw": features.DRAW_CLASS,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class HoldemSolverConfig:
    """Everything that defines a trained abstract HU Hold'em strategy."""

    encoder: str = PRIMARY_SOLVER_ENCODER
    bet_fractions: Tuple[Tuple[str, float], ...] = (("b33", 0.33), ("b75", 0.75), ("b150", 1.5))
    starting_stack: float = 100.0
    raise_cap: int = 3
    sampling: str = "external-sampling-mccfr"
    # Card buckets are the deterministic features of abstraction/features.py
    # (no equity range is used to bucket). The reference range below is used
    # only offline, for abstraction-quality metrics.
    reference_range: str = "uniform-random-hand (offline quality metrics only)"
    notes: Dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.encoder not in _ENCODERS:
            raise ValueError(f"unknown encoder {self.encoder!r}")
        if self.sampling not in SAMPLING_VARIANTS:
            raise ValueError(f"unsupported sampling variant {self.sampling!r}")
        if self.raise_cap < 1 or self.starting_stack <= 0 or not self.bet_fractions:
            raise ValueError("invalid tree parameters")

    # -- derived ----------------------------------------------------------

    def build_game(self):
        from .games.holdem import HoldemGame

        game = HoldemGame(starting_stack=self.starting_stack,
                          bet_fractions=dict(self.bet_fractions),
                          raise_cap=self.raise_cap,
                          encoder=make_encoder(self.encoder))
        game.solver_config = self
        return game

    def build_solver(self, seed: int):
        from .solvers.mccfr import MCCFRSolver

        return MCCFRSolver(self.build_game(), seed=seed)

    @property
    def perfect_recall(self) -> bool:
        return bool(getattr(make_encoder(self.encoder), "perfect_recall_by_design",
                            self.encoder == "bucket"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("notes")
        d["bet_fractions"] = [list(x) for x in self.bet_fractions]
        game = self.build_game()
        d["config_version"] = CONFIG_VERSION
        d["game_signature"] = game.signature()
        d["encoder_signature"] = game.encoder_signature()
        d["card_feature_tables"] = card_feature_tables_digest()
        d["perfect_recall"] = self.perfect_recall
        return d

    def signature(self) -> str:
        """``HoldemSolverConfig:v1:<sha256 prefix>`` over :meth:`to_dict`."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return f"HoldemSolverConfig:v{CONFIG_VERSION}:{hashlib.sha256(blob.encode()).hexdigest()[:20]}"


PRIMARY_CONFIG = HoldemSolverConfig()
