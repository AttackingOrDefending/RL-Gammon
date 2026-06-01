"""Integration tests of the agent, search, and MuZero stack on the real OpenSpiel engine (skipped if absent)."""

import numpy as np
import pytest
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.game import PossibleEngine, apply_sampled_chance, board_features, create_game
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.openspiel_adapter import is_openspiel_available
from rlgammon.models.value_model import TDGammonNet
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator

pytestmark = pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")

TINY_SIMULATIONS = 4


def _root_decision_state() -> GameState:
    """
    Return a real backgammon state with the opening roll resolved (a decision node).

    :return: a decision-node state for the player to move
    """
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    apply_sampled_chance(state, np.random.default_rng(0))
    return state


def test_td_agent_choose_move_real() -> None:
    """Test that the TD agent picks a legal action on a real backgammon state."""
    state = _root_decision_state()
    action = TDAgent().choose_move(state.legal_actions(), state)
    assert action in state.legal_actions()


def test_star_minimax_real() -> None:
    """Test that StarMinimax returns a legal action at depth 1 on a real state."""
    state = _root_decision_state()
    planner = StarMinimax(ValueNetEvaluator(TDGammonNet()), max_depth=1)
    result = planner.search(state)
    assert result.best_action in state.legal_actions()


def test_muzero_mcts_real_root() -> None:
    """Test that the MuZero MCTS produces visit counts over the legal actions of a real state."""
    state = _root_decision_state()
    config = MuZeroConfig(
        state_channels=16,
        hidden_sizes=(16,),
        codebook_size=2,
        value_support_size=3,
        reward_support_size=3,
        num_simulations=TINY_SIMULATIONS,
    )
    network = StochasticMuZeroNetwork(config)
    mcts = StochasticMuZeroMCTS(config, network, np.random.default_rng(0))
    observation = th.tensor(board_features(state, state.current_player()), dtype=th.float32).unsqueeze(0)
    visits = mcts.run(observation, state.legal_actions(), add_exploration_noise=False)
    assert sum(visits.values()) == config.num_simulations
    assert set(visits).issubset(set(state.legal_actions()))
