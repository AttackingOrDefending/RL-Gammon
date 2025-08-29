"""Testing class with a random agent."""
import numpy as np
import pyspiel  # type: ignore[import-not-found]

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.rlgammon_types import BLACK, WHITE
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
        wins = 0
        draws = 0
        losses = 0
        points_white = 0
        points_black = 0
        env = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
        agent.set_color(WHITE)
        self.testing_agent.set_color(BLACK)
        for _test_game in range(self.episodes_in_test):
            state = env.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes, strict=False)
                    action = np.random.choice(action_list, p=prob_list)
                    state.apply_action(action)
                else:
                    # Get current player
                    current_player = state.current_player()

                    # Get legal actions
                    legal_actions = state.legal_actions()

                    if current_player == agent.color:
                        action, _ = agent.choose_move(legal_actions, state)
                    else:
                        action, _ = self.testing_agent.choose_move(legal_actions, state)

                    # Apply action
                    state.apply_action(action)

            rewards = state.returns()
            if (agent.color == WHITE and rewards[WHITE] > 0) or (agent.color == BLACK and rewards[BLACK] > 0):
                wins += 1
                points_white += rewards[agent.color]
            else:
                losses += 1
                opponent_color = WHITE if agent.color == BLACK else BLACK
                points_black += rewards[opponent_color]

            agent.flip_color()
            self.testing_agent.flip_color()

        return {"win_rate": wins / self.episodes_in_test,
                "draws": draws / self.episodes_in_test,
                "losses": losses / self.episodes_in_test,
                "points_white": points_white / self.episodes_in_test,
                "points_black": points_black / self.episodes_in_test}
