"""Implementation of no exploration."""
from rlgammon.exploration import BaseExploration
from rlgammon.rlgammon_types import ActionGNU, ActionSetGNU


class NoExploration(BaseExploration):
    """Class implementing no exploration."""

    def should_explore(self) -> bool:
        """Exploration should never occur so always returns False."""
        return False

    def explore(self, actions: list[int] | ActionSetGNU) -> int | ActionGNU:
        """Exploration is not allowed, so raise an error, if it's attempted to run."""
        raise NotImplementedError

    def update(self) -> None:
        """Updating the exploration doesn't do anything, there is nothing to update."""
