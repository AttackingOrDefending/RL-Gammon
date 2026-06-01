"""Test the feature extractor and chance-handling helpers against the mock game."""

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.feature_extractor import (
    N_BOARD_FEATURES,
    apply_sampled_chance,
    board_features,
    chance_action_probs,
    features_side_to_move,
    sample_chance,
)
from rlgammon.game.mock_game import MockGame
from rlgammon.rlgammon_types import WHITE

PROBABILITY_TOLERANCE = 1e-9


def _decision_state() -> GameState:
    """
    Return a mock decision-node state (the opening roll resolved).

    :return: a state where it is a player's turn to move
    """
    state = MockGame().new_initial_state()
    state.apply_action(2)
    return state


def test_board_features_length() -> None:
    """Test that the board features drop the dice and keep the 198 board entries."""
    state = _decision_state()
    assert len(board_features(state, WHITE)) == N_BOARD_FEATURES


def test_features_side_to_move_matches_current_player() -> None:
    """Test that the side-to-move features equal the current player's board features."""
    state = _decision_state()
    assert features_side_to_move(state) == board_features(state, state.current_player())


def test_chance_action_probs_split() -> None:
    """Test that the chance helper splits outcomes into parallel action and probability lists."""
    state = MockGame().new_initial_state()
    actions, probs = chance_action_probs(state)
    assert len(actions) == len(probs)
    assert abs(sum(probs) - 1.0) < PROBABILITY_TOLERANCE


def test_sample_chance_is_deterministic_with_seed() -> None:
    """Test that sampling a chance outcome is reproducible under a fixed seed."""
    state = MockGame().new_initial_state()
    first = sample_chance(state, np.random.default_rng(0))
    second = sample_chance(state, np.random.default_rng(0))
    assert first == second


def test_apply_sampled_chance_resolves_chance_node() -> None:
    """Test that applying a sampled chance turns a chance node into a decision node."""
    state = MockGame().new_initial_state()
    assert state.is_chance_node()
    applied = apply_sampled_chance(state, np.random.default_rng(1))
    assert not state.is_chance_node()
    assert applied in [action for action, _ in MockGame().new_initial_state().chance_outcomes()]
