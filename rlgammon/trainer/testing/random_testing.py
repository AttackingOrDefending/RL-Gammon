"""Testing class with a random agent."""

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.trainer.testing.base_testing import BaseTesting


class RandomTesting(BaseTesting):
    """Testing class, where agents are tested against a random agent."""

    def __init__(self, episodes_in_test: int) -> None:
        """
        Constructor for RandomTesting, that initializes the random agent,
        and stores the specified number of episodes in each test.

        :param episodes_in_test: test episodes to be run in each test
        """
        self.episodes_in_test = episodes_in_test
        self.testing_agent = RandomAgent()

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """
        Test the provided agent against a random agent, for the number of episodes specified in the constructor.

        :param agent: agent to be tested
        :return: results of test, with win, draw, and loss rate recorded (as fractions)
        """
        wins = 0
        draws = 0
        losses = 0
        env = BackgammonEnv()
        agent_player = 1
        for _test_game in range(self.episodes_in_test):
            env.reset()
            done = False
            trunc = False
            reward = 0.0
            while not done and not trunc:
                action = agent.choose_move(env) if env.current_player == agent_player else self.testing_agent.choose_move(env)
                reward, done, trunc, _ = env.step(action)
            if reward == 0:
                draws += 1
            elif env.has_lost(agent_player):
                losses += 1
            else:
                wins += 1
            agent_player *= -1
        return {"win_rate": wins / self.episodes_in_test,
                "draws": draws / self.episodes_in_test,
                "losses": losses / self.episodes_in_test}
