"""Base class for testing agents."""

from rlgammon.agents.base_agent import BaseAgent


class BaseTesting:
    """Base class for testing agents."""

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """
        Test an agent and return the results of this test.

        :param agent: agent to be tested
        :return: results of test
        """
        raise NotImplementedError
