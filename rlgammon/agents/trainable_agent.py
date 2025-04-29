"""Base class for all trainable agents in the backgammon game."""
from uuid import UUID

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment.backgammon_env import BackgammonEnv


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    def update_weights(self, state_value: float, next_state_value: float) -> int:
        """
        Train the agent from the given buffer.

        :param state_value: the value of the current state
        :param next_state_value: the value of the next state
        :return: loss associated with update
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
