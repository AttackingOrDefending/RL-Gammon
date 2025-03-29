"""Base class for all agents in the backgammon game."""

from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class BaseAgent:
    """Base class for all agents in the backgammon game."""

    def choose_move(self, board: BackgammonEnv) -> tuple[int, MovePart] | None:
        """
        Chooses a move to make given the current board and dice roll.

        :param board: The current board state.
        :return: The chosen move to make.
        """
        raise NotImplementedError

    def choose_move_deprecated(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """
        Chooses a move to make given the current board and dice roll.

        :param board: The current board state.
        :param dice: The current dice roll.
        :return: The chosen move to make.
        """
        raise NotImplementedError
