"""Type aliases for exploration algorithms."""

from enum import Enum


class PossibleExploration(Enum):
    """Enumeration of possible exploration types."""

    EPSILON_GREEDY = "EG"
