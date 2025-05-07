"""Testing class with a gnubg agent."""
from rlgammon.agents.gnu_agent import GNUAgent
from rlgammon.agents.td_agent_gnu import TDAgentGnu  # type: ignore[attr-defined]
from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.gnubg.gnubg_backgammon import (  # type: ignore[attr-defined]
    GnubgEnv,
    GnubgInterface,
    evaluate_vs_gnubg,
)
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.base_testing import BaseTesting


class GNUTesting(BaseTesting):
    """Testing class, where agents are tested against a gnubg agent."""

    def __init__(self, episodes_in_test: int) -> None:
        """
        Constructor for GNUTesting, that stores the GNU interface and
        the specified number of episodes in each test.

        Note: GNU is always run on localhost at port 8001!
        :param episodes_in_test: test episodes to be run in each test
        """
        self.episodes_in_test = episodes_in_test
        self.gnu_interface = GnubgInterface("localhost", 8001)

    def configure_agent(self, agent: TrainableAgent) -> GNUAgent | None:
        """
        Create a GNU compatible agent from the provided trainable agent.
        NOTE: NEED MORE WORK TO MAKE GENERALIZABLE.

        :param agent: agent to test against GNU
        :return: GNU compatible agent derived from given agent
        """
        agent_gnu = TDAgentGnu(self.gnu_interface, None, gamma=0.99, color=agent.color)
        if not agent.get_model():
            return None
        agent_gnu.model = agent.get_model()
        return agent_gnu  # type: ignore[no-any-return]

    def test(self, agent: TrainableAgent) -> dict[str, float]:
        """
        Test the provided agent against a random agent, for the number of episodes specified in the constructor.

        :param agent: agent to be tested
        :return: results of test, with win, draw, and loss rate recorded (as fractions)
        """
        # Note: only white tested
        agent_gnu = self.configure_agent(agent)

        if not agent_gnu:
            msg = "The agent needs a model to be tested against GNU!"
            raise ValueError(msg)

        wins = 0
        draws = 0
        losses = 0
        agent_gnu.set_color(WHITE)
        for _ in range(self.episodes_in_test):
            gnu_env = GnubgEnv(self.gnu_interface)
            results = evaluate_vs_gnubg(agent_gnu, gnu_env, 1)
            wins += results[WHITE]
            losses += results[BLACK]

        return {"win_rate": wins / self.episodes_in_test,
                "draws": draws / self.episodes_in_test,
                "losses": losses / self.episodes_in_test}
