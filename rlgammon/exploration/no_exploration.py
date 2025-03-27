"""Implementation of no exploration."""
from rlgammon.environment import BackgammonEnv
from rlgammon.exploration import BaseExploration
from rlgammon.rlgammon_types import MovePart


class NoExploration(BaseExploration):
    """Class implementing no exploration."""

    def should_explore(self) -> bool:
        """Exploration should never occur so always returns False."""
        return False

    def explore(self, valid_actions: list[tuple[BackgammonEnv, tuple[int, MovePart]]]) -> tuple[int, MovePart]:
        """Exploration is not allowed, so raise an error, if it's attempted to run."""
        raise NotImplementedError

    def update(self) -> None:
        """Updating the exploration doesn't do anything, there is nothing to update."""
