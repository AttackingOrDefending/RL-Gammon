"""TODO."""
from rlgammon.agents.base_agent import BaseAgent
from rlgammon.trainer.testing.base_testing import BaseTesting


class GNUTesting(BaseTesting):
    def __init__(self) -> None:
        pass

    def test(self, agent: BaseAgent) -> dict[str, float]:
        pass
