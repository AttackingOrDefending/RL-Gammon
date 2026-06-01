"""Testing class with a random agent."""
import numpy as np

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.planning_agent import PlanningAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.game import PossibleEngine, create_game
from rlgammon.models.base_model import BaseModel
from rlgammon.planning.agent_builder import build_planning_agent
from rlgammon.planning.planning_types import SearchConfig
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.base_testing import BaseTesting


class RandomTesting(BaseTesting):
    """Testing class, where agents are tested against a random agent."""

    def __init__(self, episodes_in_test: int, color: int = WHITE,
                 eval_search: SearchConfig | None = None) -> None:
        """
        Constructor for RandomTesting, that initializes the random agent,
        and stores the specified number of episodes in each test.

        :param episodes_in_test: test episodes to be run in each test
        :param color: 0 or 1 representing which player the random opponent controls
        :param eval_search: optional search config; when set, the agent-under-test "thinks deeper" at
            test time by routing its moves through a planning agent built from its trained model
        """
        self.episodes_in_test = episodes_in_test
        self.testing_agent = RandomAgent(color)
        self.eval_search = eval_search

    def _build_eval_agent(self, agent: BaseAgent) -> PlanningAgent | None:
        """
        Build the deeper-thinking planning agent for ``agent`` if an evaluation search is configured.

        :param agent: the agent under test (must expose a value model via ``get_model``)
        :return: a planning agent wrapping the agent's trained model, or ``None`` for 1-ply testing
        """
        if self.eval_search is None or not isinstance(agent, TrainableAgent):
            return None
        model = agent.get_model()
        if not isinstance(model, BaseModel):
            return None
        return build_planning_agent(model, self.eval_search, color=agent.color)

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """
        Test the provided agent against a random agent, for the number of episodes specified in the constructor.

        :param agent: agent to be tested
        :return: results of test, with win, draw, and loss rate recorded (as fractions)
        """
        wins = 0
        draws = 0
        losses = 0
        points_white = 0.0
        points_black = 0.0
        game = create_game(PossibleEngine.OPEN_SPIEL)
        eval_agent = self._build_eval_agent(agent)
        agent.set_color(WHITE)
        self.testing_agent.set_color(BLACK)
        for _test_game in range(self.episodes_in_test):
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes, strict=False)
                    state.apply_action(int(np.random.choice(action_list, p=prob_list)))
                else:
                    # Get current player
                    current_player = state.current_player()

                    # Get legal actions
                    legal_actions = state.legal_actions()

                    if current_player == agent.color:
                        # Route the tested agent through the deeper search when one is configured.
                        moving_agent: BaseAgent = eval_agent if eval_agent is not None else agent
                        if eval_agent is not None:
                            eval_agent.set_color(agent.color)
                        action = moving_agent.choose_move(legal_actions, state)
                    else:
                        action = self.testing_agent.choose_move(legal_actions, state)

                    # Apply action (action ids are plain ints in the OpenSpiel engine)
                    state.apply_action(action)  # type: ignore[arg-type]

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
