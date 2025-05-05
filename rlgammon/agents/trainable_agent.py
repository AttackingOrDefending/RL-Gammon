"""Base class for all trainable agents in the backgammon game."""
from abc import abstractmethod
from uuid import UUID

import torch as th

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.rlgammon_types import State


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    @abstractmethod
    def train(self, state: State, next_state: State, reward: int, done: bool) -> float:
        """
        Train the agent from the given buffer.

        :param state: the current state
        :param next_state: the next state
        :param reward: the reward from the environment associated with the current state
        :param done: whether the state is the last one of the round
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
