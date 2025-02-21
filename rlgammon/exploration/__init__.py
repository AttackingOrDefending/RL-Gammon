"""Init file for exploration algorithms."""

from rlgammon.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration
from rlgammon.exploration.base_exploration import BaseExploration

__all__ = ["EpsilonGreedyExploration", "BaseExploration"]
