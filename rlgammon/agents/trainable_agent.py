"""Base class for all trainable agents in the backgammon game."""
from abc import abstractmethod
from uuid import UUID

import torch as th

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.rlgammon_types import Features


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    @abstractmethod
    def evaluate_position(self, state: Features, decay: bool = False) -> th.Tensor:
        """
        Evaluate the given position.

        :param state: state to evaluate
        :param decay: flag whether to decay the value or not
        :return: th tensor storing the value of the provided state
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, p: th.Tensor, p_next: th.Tensor) -> float:
        """
        Train the agent from the given buffer.

        :param p: value of current state
        :param p_next: value of the next state or final reward if terminal state
        :return: loss associated with update
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str | None = None) -> None:
        """
        Save the agent model.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        :param main_filename: name of the file under which the agent is to be saved
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, agent_main_filename: str) -> th.nn.Module:
        """
        Load the agent model.

        :param agent_main_filename: name of the file under which the agent is saved
        :return: the loaded agent model
        """
        raise NotImplementedError
