"""Generic agents + reproducible tournaments for hidden-information games."""

from .adapters import (BestResponseAgent, PolicyTableAgent, RandomAgent,
                       ScriptedKuhnAgent, kuhn_policy_agent)
from .base import Agent
from .tournament import MatchResult, play_episode, play_match, round_robin

__all__ = ["Agent", "RandomAgent", "PolicyTableAgent", "BestResponseAgent",
           "ScriptedKuhnAgent", "kuhn_policy_agent", "MatchResult",
           "play_episode", "play_match", "round_robin"]
