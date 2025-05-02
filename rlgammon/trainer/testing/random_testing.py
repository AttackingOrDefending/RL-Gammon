"""Testing class with a random agent."""

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import WHITE
from rlgammon.trainer.testing.base_testing import BaseTesting


class RandomTesting(BaseTesting):
    """Testing class, where agents are tested against a random agent."""

    def __init__(self, episodes_in_test: int, color: int = WHITE) -> None:
        """
        Constructor for RandomTesting, that initializes the random agent,
        and stores the specified number of episodes in each test.

        :param episodes_in_test: test episodes to be run in each test
        """
        self.episodes_in_test = episodes_in_test
        self.testing_agent = RandomAgent(color)

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """
        Test the provided agent against a random agent, for the number of episodes specified in the constructor.

        :param agent: agent to be tested
        :return: results of test, with win, draw, and loss rate recorded (as fractions)
        """
        # TODO CHECK COLORS

        wins = 0
        draws = 0
        losses = 0
        env = BackgammonEnv()
        agent_player = WHITE
        for _test_game in range(self.episodes_in_test):
            agent_color, first_roll, state = env.reset()
            done = False
            reward = 0.0
            while not done:
                # If this is the first step, take the roll from env, else roll yourself
                if first_roll:
                    roll = first_roll
                    first_roll = None
                else:
                    roll = agent.roll_dice()

                actions = env.get_valid_actions(roll)
                action = agent.choose_move(actions, env) \
                    if env.current_player == agent_player else self.testing_agent.choose_move(actions, env)
                next_state, reward, done, winner = env.step(action)
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
