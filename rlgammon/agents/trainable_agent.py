"""Base class for all trainable agents in the backgammon game."""
from abc import abstractmethod
from uuid import UUID

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import State


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    @abstractmethod
    def train(self, state: State, next_state: State, reward: int, done: bool) -> float:
        """
        Train the agent from the given buffer.

        :param state: the current state
        :param next_state: the next state
        :param done: whether the state is the last one of the round
        :return: loss associated with update
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the cache of the evaluate_position method."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_position(self, board: BackgammonEnv) -> float:
        """
        Evaluate the position of the current board state.

        :param board: The current board state
        :return: value of the current position
        """
        raise NotImplementedError
