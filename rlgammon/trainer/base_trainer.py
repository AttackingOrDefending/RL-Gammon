"""Base trainer class for all trainers used for training rl-algorithms."""
from abc import abstractmethod
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pyspiel  # type: ignore[import-not-found]
from pyspiel import BackgammonState

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers import BaseBuffer, UniformBuffer
from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.exploration import BaseExploration, EpsilonGreedyExploration
from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.exploration.no_exploration import NoExploration
from rlgammon.trainer.logger.logger import Logger
from rlgammon.trainer.testing.base_testing import BaseTesting
from rlgammon.trainer.testing.gnu_testing import TDGNUTesting
from rlgammon.trainer.testing.random_testing import RandomTesting
from rlgammon.trainer.testing.testing_types import PossibleTesting
from rlgammon.trainer.trainer_errors.trainer_errors import (
    WrongBufferTypeError,
    WrongExplorationTypeError,
    WrongTestingTypeError,
)
from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid
from rlgammon.trainer.trainer_types import TrainerType


class BaseTrainer:
    """Base trainer class for all trainers used for training rl-algorithms."""

    def __init__(self) -> None:
        """Constructor for the BaseTrainer containing the parameters for the trainer."""
        self.parameters: dict[str, Any] = {}

    def create_buffer_from_parameters(self, game: pyspiel.Game) -> BaseBuffer:
        """
        Create a new buffer of the type provided in the parameters.

        :param game: the pyspiel game instance
        :return: buffer of the type provided in the parameters
        """
        match self.parameters["buffer"]:
            case PossibleBuffers.UNIFORM:
                buffer = UniformBuffer(game.num_distinct_actions(), self.parameters["buffer_capacity"])
            case _:
                raise WrongBufferTypeError
        return buffer

    def create_logger_from_parameters(self, training_session_id: UUID, trainer_type: TrainerType) -> Logger:
        """
        Create a new logger, either an empty one just initialized
        or one loaded with the logger name provided in the parameters.

        :return: new logger
        """
        logger = Logger(training_session_id, trainer_type)
        if self.parameters["load_logger"]:
            logger.load(self.parameters["logger_name"])
        return logger

    def create_testing_from_parameters(self) -> BaseTesting:
        """
        Create a new testing object of the type provided in the parameters.

        :return: testing object of the type provided in the parameters
        """
        match self.parameters["testing_type"]:
            case PossibleTesting.RANDOM:
                testing = RandomTesting(self.parameters["episodes_in_test"])
            case PossibleTesting.TD_GNU:
                testing = TDGNUTesting(self.parameters["episodes_in_test"])  # type: ignore[assignment]
            case _:
                raise WrongTestingTypeError
        return testing

    def create_explorer_from_parameters(self) -> BaseExploration:
        """
        Create a new exploration algorithm of the type provided in the parameters.

        :return: exploration algorithm of the type provided in the parameters
        """
        match self.parameters["exploration"]:
            case PossibleExploration.EPSILON_GREEDY:
                explorer: EpsilonGreedyExploration | NoExploration = (
                    EpsilonGreedyExploration(self.parameters["start_epsilon"], self.parameters["end_epsilon"],
                                             self.parameters["update_decay"], self.parameters["steps_per_update"]))
            case PossibleExploration.NO_EXPLORATION:
                explorer = NoExploration()
            case _:
                raise WrongExplorationTypeError
        return explorer

    def is_ready_for_training(self) -> bool:
        """Checks if the parameters have been loaded, which indicates whether trainer is ready."""
        return bool(self.parameters)

    def load_parameters(self, json_parameters_name: str) -> None:
        """
        Load parameters to be used for training, and verify their validity.

        :param json_parameters_name: name of the json parameters file
        :raises: ValueError: the parameters are invalid, i.e. don't contain some data, or have invalid types
        """
        parameter_file_path = Path(__file__).parent
        parameter_file_path = parameter_file_path.joinpath("trainer_parameters/parameters/")
        path = parameter_file_path.joinpath(json_parameters_name)

        with path.open() as json_parameters:
            parameters = json.load(json_parameters)

        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            msg = "Invalid parameters"
            raise ValueError(msg)

    @abstractmethod
    def train(self, agent: TrainableAgent) -> None:
        """
        Train the provided agent.

        :param agent: agent to be trained
        """
        raise NotImplementedError
