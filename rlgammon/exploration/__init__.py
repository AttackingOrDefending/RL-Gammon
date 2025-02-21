"""Init file for exploration algorithms."""

from rlgammon.exploration.base_exploration import BaseExploration
from rlgammon.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration

__all__ = ["BaseExploration", "EpsilonGreedyExploration"]
