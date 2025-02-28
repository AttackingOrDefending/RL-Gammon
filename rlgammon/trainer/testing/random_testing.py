"""TODO."""

from rlgammon.agents.random_agent import RandomAgent
from rlgammon.trainer.testing.base_testing import BaseTesting


class RandomTesting(BaseTesting):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        self.agent = RandomAgent()
