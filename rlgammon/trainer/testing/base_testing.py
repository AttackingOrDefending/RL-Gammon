from rlgammon.agents.base_agent import BaseAgent


class BaseTesting:
    """TODO."""

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """TODO."""
        raise NotImplementedError
