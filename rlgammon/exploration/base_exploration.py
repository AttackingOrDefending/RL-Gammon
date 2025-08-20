"""Base class for all exploration algorithms."""

from abc import abstractmethod

from rlgammon.rlgammon_types import ActionInfoTuple, ActionSetGNU


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
    def explore(self, actions: list[int] | ActionSetGNU) -> ActionInfoTuple:
        """
        Method to conduct exploration.

        :param actions: all valid actions from the current state
        :return: the final action to execute
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Update the exploration algorithm."""
        raise NotImplementedError
