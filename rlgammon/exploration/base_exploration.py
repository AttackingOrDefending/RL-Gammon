"""Base class for all exploration algorithms."""

from abc import abstractmethod

from rlgammon.rlgammon_types import MovePart


class BaseExploration:
    """Base class for all exploration algorithms."""

    @abstractmethod
    def explore(self, action: list[tuple[int, MovePart]], valid_actions: list[list[tuple[int, MovePart]]],
                ) -> list[tuple[int, MovePart]]:
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
