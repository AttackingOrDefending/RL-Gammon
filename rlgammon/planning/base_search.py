"""Base class for all search/planning algorithms."""

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.planning_types import Evaluator, SearchResult


class BaseSearch:
    """Base class for all search/planning algorithms."""

    def __init__(self, evaluator: Evaluator, max_depth: int) -> None:
        """
        Construct the base search with a pluggable leaf evaluator and a maximum depth.

        :param evaluator: the leaf evaluator used to score non-terminal frontier states
        :param max_depth: the maximum search depth in decision plies
        """
        self._evaluator = evaluator
        self._max_depth = max_depth

    def search(self, state: GameState, deadline: float | None = None) -> SearchResult:
        """
        Run the search from the given root state and return its result.

        :param state: the root game state (a decision node) to search from
        :param deadline: an optional ``time.monotonic()`` timestamp to stop by (``None`` = fixed behaviour)
        :return: the search result with the chosen action, its value and search statistics
        """
        raise NotImplementedError
