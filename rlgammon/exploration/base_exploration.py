"""Base class for all exploration algorithms."""

from abc import abstractmethod

from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class BaseExploration:
    """Base class for all exploration algorithms."""

    @abstractmethod
    def should_explore(self) -> bool:
        """
        Method to determine whether to explore this time step.

        :return: boolean, indicating whether to explore this time step
        """
        raise NotImplementedError

    @abstractmethod
    def explore(self, valid_actions: list[tuple[BackgammonEnv, tuple[int, MovePart]]]) -> tuple[int, MovePart]:
        """
        Method to conduct exploration.

        :param valid_actions: all valid actions from the current state
        :return: the final action to execute
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Update the exploration algorithm."""
        raise NotImplementedError
