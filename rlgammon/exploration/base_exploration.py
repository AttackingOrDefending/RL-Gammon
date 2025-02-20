"""Base class for all exploration algorithms."""

from abc import abstractmethod


class BaseExploration:
    """Base class for all exploration algorithms."""

    @abstractmethod
    def explore(self, action: int, valid_actions: list[int]) -> int:
        """
        Method to conduct exploration.

        :param action: current action chosen by the agent
        :param valid_actions: all valid actions from the current state
        :return: the final action to execute
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Update the exploration algorithm."""
        raise NotImplementedError
