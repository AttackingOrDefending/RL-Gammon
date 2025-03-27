"""Base class for all trainable agents in the backgammon game."""
from uuid import UUID

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    def choose_move(self, board: BackgammonEnv) -> tuple[int, MovePart]:
        """
        Chooses a move to make given the current board and dice roll.

        :param board: The current board state
        :return: The chosen move to make
        """
        raise NotImplementedError

    def choose_move_deprecated(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """
        Chooses a move to make given the current board and dice roll.

        :param board: The current board state
        :param dice: The current dice roll
        :return: The chosen move to make
        """
        raise NotImplementedError

    def train(self, buffer: BaseBuffer) -> None:
        """
        Train the agent from the given buffer.

        :param buffer: buffer with stored observations
        """
        raise NotImplementedError

    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str | None = None,
             target_filename: str | None = None, optimizer_filename: str | None = None) -> None:
        """
        Save the agent.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        :param main_filename: filename where the main network is to be saved
        :param target_filename: filename where the target network is to be saved
        :param optimizer_filename: filename where the optimizer is to be saved
        """
        raise NotImplementedError

    def clear_cache(self) -> None:
        """Clear the cache of the evaluate_position method."""
        raise NotImplementedError

    def evaluate_position(self, board: BackgammonEnv) -> float:
        """
        Evaluate the position of the current board state.

        :param board: The current board state
        :return: value of the current position
        """
        raise NotImplementedError
