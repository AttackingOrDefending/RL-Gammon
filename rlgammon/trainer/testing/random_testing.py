"""Testing class with a random agent."""

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.base_testing import BaseTesting

# TODO CHECK IF CORRECT

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
        agent.set_color(WHITE)
        self.testing_agent.set_color(BLACK)
        for _test_game in range(self.episodes_in_test):
            agent_color, first_roll, state = env.reset()
            done = False
            winner = None
            while not done:
                # If this is the first step, take the roll from env, else roll yourself
                if first_roll:
                    roll = first_roll
                    first_roll = None
                elif env.current_agent == agent.color:
                    roll = agent.roll_dice()
                else:
                    roll = self.testing_agent.roll_dice()

                actions = env.get_valid_actions(roll)
                action = agent.choose_move(actions, env) \
                    if env.current_agent == agent.color else self.testing_agent.choose_move(actions, env)
                next_state, reward, done, winner = env.step(action)
                state = next_state

            if winner == agent.color:
                wins += 1
            elif not winner:
                draws += 1
            else:
                losses += 1

            agent.flip_color()
            self.testing_agent.flip_color()

        return {"win_rate": wins / self.episodes_in_test,
                "draws": draws / self.episodes_in_test,
                "losses": losses / self.episodes_in_test}
