"""Base trainer class for all trainers used for training rl-algorithms."""

from abc import abstractmethod
from typing import Any

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers import BaseBuffer, UniformBuffer
from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.environment import BackgammonEnv
from rlgammon.exploration import BaseExploration, EpsilonGreedyExploration
from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.rlgammon_types import Input, MoveList
from rlgammon.trainer.trainer_errors.trainer_errors import WrongBufferTypeError, WrongExplorationTypeError


class BaseTrainer:
    """Base trainer class for all trainers used for training rl-algorithms."""

    def __init__(self) -> None:
        """Constructor for the BaseTrainer containing the parameters for the trainer."""
        self.parameters: dict[str, Any] = {}

    def finalize_data(self, episode_buffer: list[tuple[Input, Input, MoveList, bool, int]],
                      losing_player: int, final_reward: float, buffer: BaseBuffer) -> None:
        """
        Finalize the data by updating the rewards for each time step
        to take into account the loss of value of rewards closer to the start of the game, and
        to set the reward negative for the losing player.

        :param episode_buffer: the data from the completed episode
        :param losing_player: the player who lost the game
        :param final_reward: the reward given at the end of the game
        :param buffer: buffer to which to add the data
        """
        for i, (state, next_state, action, done, player) in enumerate(reversed(episode_buffer)):
            reward = final_reward * self.parameters["decay"] ** i
            if player == losing_player:
                reward *= -1
            buffer.record(state, next_state, action, reward, done)

    def create_buffer_from_parameters(self, env: BackgammonEnv) -> BaseBuffer:
        """
        Create a new buffer of the type provided in the parameters.

        :return: buffer of the type provided in the parameters
        """
        if self.parameters["buffer"] == PossibleBuffers.UNIFORM:
            buffer = UniformBuffer(env.observation_shape, env.action_shape, self.parameters["buffer_capacity"])
        else:
            raise WrongBufferTypeError

        return buffer

    def create_explorer_from_parameters(self) -> BaseExploration:
        """
        Create a new exploration algorithm of the type provided in the parameters.

        :return: exploration algorithm of the type provided in the parameters
        """
        if self.parameters["exploration"] == PossibleExploration.EPSILON_GREEDY:
            explorer = EpsilonGreedyExploration(self.parameters["start_epsilon"], self.parameters["end_epsilon"],
                                                self.parameters["update_decay"], self.parameters["steps_per_update"])
        else:
            raise WrongExplorationTypeError

        return explorer

    def is_ready_for_training(self) -> bool:
        """Checks if the parameters have been loaded, which indicates whether trainer is ready."""
        return self.parameters == {}

    @abstractmethod
    def load_parameters(self, json_parameters_name: str) -> None:
        """
        Load parameters to be used for training.

        :param json_parameters_name: name of the json parameters file
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, agent: TrainableAgent) -> None:
        """
        Train the provided agent.

        :param agent: agent to be trained
        """
        raise NotImplementedError
