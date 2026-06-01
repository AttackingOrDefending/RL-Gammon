"""Tests for the planning agent wired to a star-minimax planner on the mock game."""

from rlgammon.agents.planning_agent import PlanningAgent
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.rlgammon_types import WHITE


class ConstantEvaluator:
    """A leaf evaluator returning a constant value for every non-terminal state."""

    def __init__(self, value: float = 0.0) -> None:
        """
        Construct the constant evaluator.

        :param value: the constant equity to return for any state
        """
        self._value = value

    def evaluate(self, state: GameState, perspective: int) -> float:  # noqa: ARG002
        """
        Return the constant value regardless of the state or perspective.

        :param state: the game state to evaluate (unused)
        :param perspective: the player whose equity to return (unused)
        :return: the configured constant value
        """
        return self._value


def test_agent_picks_winning_action() -> None:
    """Test that a planning agent with a star-minimax planner picks the winning action."""
    state = MockGame.contrived_win_in_one(WHITE)
    planner = StarMinimax(ConstantEvaluator(0.0), max_depth=1)
    agent = PlanningAgent(planner, color=WHITE)
    agent.episode_setup()
    assert agent.choose_move(state.legal_actions(), state) == MockGame.winning_action()
