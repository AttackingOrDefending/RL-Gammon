"""Type aliases, the search result container, and the leaf-evaluator protocol for planning."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from rlgammon.game.backgammon_protocol import GameState


@dataclass(frozen=True)
class SearchResult:
    """Immutable summary of a single search from a root decision node."""

    best_action: int
    value: float
    nodes_visited: int
    pv: list[int]


class PossibleSearch(Enum):
    """Enumeration of possible search/planning algorithms."""

    STAR_MINIMAX = "SM"
    MCTS = "MCTS"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleSearch":
        """
        Convert a string, found e.g. in JSON parameters, to a PossibleSearch enum.

        :param string_to_convert: the string value to convert
        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "SM":
                return PossibleSearch.STAR_MINIMAX
            case "MCTS":
                return PossibleSearch.MCTS
            case _:
                return None  # type: ignore[return-value]


@dataclass(frozen=True)
class SearchConfig:
    """Immutable bundle of search hyper-parameters, e.g. a separate "think deeper at test time" budget."""

    search_type: PossibleSearch
    max_depth: int = 2
    num_simulations: int = 200
    use_star2: bool = True
    c_uct: float = 1.4


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for a pluggable leaf evaluator returning a player's equity in points."""

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s equity (in points) for the given state.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the estimated equity of ``perspective`` in points
        """
