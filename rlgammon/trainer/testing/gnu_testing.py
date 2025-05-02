"""TODO."""
from rlgammon.agents.gnu_agent import GNUAgent
from rlgammon.environment.gnubg.gnubg_backgammon import GnubgEnv, GnubgInterface, evaluate_vs_gnubg
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.base_testing import BaseTesting


class GNUTesting(BaseTesting):
    """TODO."""

    def __init__(self, episodes_in_test: int) -> None:
        """
        Constructor for GNUTesting, that stores the GNU interface and
        the specified number of episodes in each test.

        Note: GNU is always run on localhost at port 8001!
        :param episodes_in_test: test episodes to be run in each test
        """
        self.episodes_in_test = episodes_in_test
        self.gnu_interface = GnubgInterface("localhost", 8001)

    def test(self, agent: GNUAgent) -> dict[str, float]:
        """
        Test the provided agent against a random agent, for the number of episodes specified in the constructor.

        :param agent: agent to be tested
        :return: results of test, with win, draw, and loss rate recorded (as fractions)
        """
        # TODO CHECK COLORS
        wins = 0
        draws = 0
        losses = 0
        color = WHITE
        for _ in range(self.episodes_in_test):
            gnu_env = GnubgEnv(self.gnu_interface)
            wins = evaluate_vs_gnubg(agent, gnu_env, 1)
            wins += wins[color]
            draws += 0 # TODO NOT POSSIBLE DRAWS
            losses += wins[BLACK if color == WHITE else WHITE]

            color = BLACK if wins else WHITE
        return {"win_rate": wins / self.episodes_in_test,
                "draws": draws / self.episodes_in_test,
                "losses": losses / self.episodes_in_test}