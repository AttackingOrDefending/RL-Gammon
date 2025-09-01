"""Base class for all trainable agents in the backgammon game."""
from abc import abstractmethod
from typing import Any
from uuid import UUID

import torch as th

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.models.model_types import ActorCriticOutput, StandardOutput
from rlgammon.rlgammon_types import Feature


class TrainableAgent(BaseAgent):
    """Base class for all trainable agents in the backgammon game."""

    @abstractmethod
    def evaluate_position(self, state: Feature, decay: bool = False) -> StandardOutput | ActorCriticOutput:
        """
        Evaluate the given position.

        :param state: state to evaluate
        :param decay: flag whether to decay the value or not
        :return: th tensor storing the value of the provided state
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, action_info: Any, reward: int, state: Feature, next_state: Feature, done: bool) -> float:
        """
        Train the agent using data generated through interactions with the environment.

        :param action_info: additional info generated when choosing moves; e.g. MCTS policy
        :param reward: reward received at the current state
        :param state: current state of the game
        :param next_state: state of the game after performing the choosen move
        :param done: boolean flag indicating whether the game is over
        :return: loss associated with the update
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str) -> None:
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

    @abstractmethod
    def get_model(self) -> th.nn.Module | None:
        """
        Get the model the agent is using, if it has one.

        :return: the agent model if it has one, else return None
        """
        raise NotImplementedError
