"""Modular search/planning package: expectiminimax (star1/star2) and stochastic UCT MCTS.

Both planners use a pluggable leaf :class:`Evaluator` and follow a negamax value convention where
every node returns a value from the mover's perspective. The package mirrors the structure of the
other algorithm packages: a ``planning_types`` module of type aliases and a ``PossibleSearch`` enum,
a ``planning_errors`` subpackage, and a ``create_search`` factory.
"""

from rlgammon.planning.base_search import BaseSearch
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.leaf_evaluator import RolloutEvaluator, ValueNetEvaluator
from rlgammon.planning.mcts import StochasticMCTS
from rlgammon.planning.planning_types import Evaluator, PossibleSearch, SearchResult
from rlgammon.planning.search_factory import create_search

__all__ = [
    "BaseSearch",
    "Evaluator",
    "PossibleSearch",
    "RolloutEvaluator",
    "SearchResult",
    "StarMinimax",
    "StochasticMCTS",
    "ValueNetEvaluator",
    "create_search",
]
