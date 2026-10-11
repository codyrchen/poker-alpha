"""Generic agent protocol for extensive-form hidden-information games.

An agent sees ONLY its information set key and the legal actions — the
same interface the solvers use — so no agent can accidentally condition
on hidden state. Adapters exist for policy tables (CFR/MCCFR output),
random play, scripted archetypes and exact best responders; a human in
the browser Arena implements the same contract on the TypeScript side.
"""

from __future__ import annotations

import random
from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    def choose_action(self, observation: str, legal_actions: List[str],
                      rng: random.Random) -> str:
        """Pick one legal action given the agent's information set key."""
        ...

    def reset(self) -> None:
        """Called before each match (clear any adaptation state)."""
        ...

    def metadata(self) -> Dict[str, str]:
        """algorithm / information used / adaptive or fixed / source /
        limitations — honest, no marketing names."""
        ...
