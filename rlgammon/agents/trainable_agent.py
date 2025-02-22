"""TODO"""
from rlgammon.agents.base_agent import BaseAgent
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class TrainableAgent(BaseAgent):
    """TODO"""

    def choose_move(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """
        Chooses a move to make given the current board and dice roll.

        :param board: The current board state.
        :param dice: The current dice roll.
        :return: The chosen move to make.
        """
        raise NotImplementedError

    def train(self, buffer: BaseBuffer) -> None:
        """
        TODO

        :param buffer:
        """
        raise NotImplementedError

    def save(self, main_filename: str | None = None, target_filename: str | None = None,
             optimizer_filename: str | None = None) -> None:
        """
        TODO

        :param main_filename:
        :param target_filename:
        :param optimizer_filename:
        """
        raise NotImplementedError

    def clear_cache(self) -> None:
        """
        TODO
        """
        raise NotImplementedError

    def evaluate_position(self, board: BackgammonEnv) -> float:
        """
        TODO

        :param board:
        :return:
        """
        raise NotImplementedError
