"""Tests for the train-vs-test search separation (SearchConfig, build_planning_agent, RandomTesting)."""

import pytest

from rlgammon.agents.planning_agent import PlanningAgent
from rlgammon.agents.td_agent import TDAgent
from rlgammon.game.mock_game import MockGame
from rlgammon.game.openspiel_adapter import is_openspiel_available
from rlgammon.models.value_model import TDGammonNet
from rlgammon.planning.agent_builder import build_planning_agent
from rlgammon.planning.planning_types import PossibleSearch, SearchConfig
from rlgammon.rlgammon_types import WHITE
from rlgammon.trainer.testing.random_testing import RandomTesting

# Number of testing episodes used for the (engine-gated) end-to-end RandomTesting run.
TEST_EPISODES = 2
# A fixed seed so the tiny value network is built deterministically.
SEED = 123


def test_build_planning_agent_picks_winning_action() -> None:
    """Test that a planning agent built from a value model picks the immediately-winning action."""
    model = TDGammonNet(hidden=16, seed=SEED)
    config = SearchConfig(PossibleSearch.STAR_MINIMAX, max_depth=1)
    agent = build_planning_agent(model, config)
    assert isinstance(agent, PlanningAgent)
    state = MockGame.contrived_win_in_one(WHITE)
    assert agent.choose_move(state.legal_actions(), state) == MockGame.winning_action()


def test_build_planning_agent_mcts_constructs() -> None:
    """Test that an MCTS-typed config builds a planning agent that returns a legal action."""
    model = TDGammonNet(hidden=16, seed=SEED)
    config = SearchConfig(PossibleSearch.MCTS, max_depth=1, num_simulations=16)
    agent = build_planning_agent(model, config)
    assert isinstance(agent, PlanningAgent)
    state = MockGame.contrived_win_in_one(WHITE)
    assert agent.choose_move(state.legal_actions(), state) in state.legal_actions()


def test_random_testing_constructs_with_eval_search() -> None:
    """Test that RandomTesting accepts an eval-search config without error."""
    config = SearchConfig(PossibleSearch.STAR_MINIMAX, max_depth=1)
    testing = RandomTesting(episodes_in_test=TEST_EPISODES, eval_search=config)
    assert testing.eval_search is config


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is required for a real testing run.")
def test_random_testing_runs_with_eval_search() -> None:
    """Test that a deeper-thinking RandomTesting run completes and reports the expected metrics."""
    agent = TDAgent(hidden=16, seed=SEED)
    config = SearchConfig(PossibleSearch.STAR_MINIMAX, max_depth=1)
    testing = RandomTesting(episodes_in_test=TEST_EPISODES, eval_search=config)
    results = testing.test(agent)
    assert set(results) == {"win_rate", "draws", "losses", "points_white", "points_black"}
