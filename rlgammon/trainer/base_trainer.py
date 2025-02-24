from abc import abstractmethod
from typing import Any

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers import BaseBuffer
from rlgammon.rlgammon_types import Input, MoveList


class BaseTrainer:
    """TODO"""
    def __init__(self) -> None:
        """TODO"""
        self.parameters: dict[str, Any] = {}

    @abstractmethod
    def finalize_data(self, episode_buffer: list[tuple[Input, Input, MoveList, float, bool, int]],
                      losing_player: int, buffer: BaseBuffer) -> None:
        """
        TODO

        :param episode_buffer:
        :param losing_player:
        :param buffer:
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, agent: TrainableAgent) -> None:
        """TODO"""
        raise NotImplementedError

    def is_ready_for_training(self) -> bool:
        """Checks if the parameters have been loaded, which indicates whether trainer is ready."""
        if self.parameters == {}:
            return False
        return True
