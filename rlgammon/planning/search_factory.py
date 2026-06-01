"""Factory for constructing a search/planning algorithm from a PossibleSearch selector."""

from rlgammon.planning.base_search import BaseSearch
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.mcts import StochasticMCTS
from rlgammon.planning.planning_types import Evaluator, PossibleSearch


class WrongSearchTypeError(Exception):
    """Class implementing the error occurring when an unknown search type is requested."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The search type you are trying to use is not available right now! "
                         "Please check 'PossibleSearch' for available search algorithms!")


def create_search(search_type: PossibleSearch, evaluator: Evaluator, max_depth: int,
                  **kwargs: object) -> BaseSearch:
    """
    Create a search algorithm of the requested type.

    :param search_type: which search algorithm to construct
    :param evaluator: the leaf evaluator the search should use
    :param max_depth: the maximum search depth in decision plies
    :param kwargs: extra keyword arguments forwarded to the chosen search constructor
    :return: a search satisfying the BaseSearch interface
    :raises SearchDepthError: if ``max_depth`` is less than 1
    :raises WrongSearchTypeError: if the search selector is not a known PossibleSearch
    """
    match search_type:
        case PossibleSearch.STAR_MINIMAX:
            return StarMinimax(evaluator, max_depth, **kwargs)  # type: ignore[arg-type]
        case PossibleSearch.MCTS:
            return StochasticMCTS(evaluator, max_depth, **kwargs)  # type: ignore[arg-type]
        case _:
            raise WrongSearchTypeError
