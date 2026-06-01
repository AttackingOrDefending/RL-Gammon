"""Test the OpenSpiel adapter against the real `pyspiel` backgammon game (skipped if absent)."""

import numpy as np
import pytest

from rlgammon.game import DEFAULT_GAME_STRING, PossibleEngine, apply_sampled_chance, create_game
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.game.feature_extractor import N_OBS
from rlgammon.game.openspiel_adapter import OpenSpielGame, is_openspiel_available

pytestmark = pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")

NUM_BACKGAMMON_ACTIONS = 1352
FULL_SCORING_RETURNS = {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}
MAX_RANDOM_PLIES = 2000


def test_adapter_satisfies_protocols() -> None:
    """Test that the OpenSpiel game and its state satisfy the runtime-checkable protocols."""
    game = create_game(PossibleEngine.OPEN_SPIEL)
    assert isinstance(game, BackgammonGame)
    assert isinstance(game.new_initial_state(), GameState)


def test_num_distinct_actions() -> None:
    """Test that full-scoring backgammon exposes the expected action-space size."""
    assert OpenSpielGame(DEFAULT_GAME_STRING).num_distinct_actions() == NUM_BACKGAMMON_ACTIONS


def test_first_node_is_chance() -> None:
    """Test that the opening node is a dice chance node."""
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    assert state.is_chance_node()


def test_observation_tensor_length() -> None:
    """Test that the observation tensor has the full backgammon length after the opening roll."""
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    apply_sampled_chance(state, np.random.default_rng(0))
    assert len(state.observation_tensor(0)) == N_OBS


def test_terminal_returns_are_full_scoring() -> None:
    """Test that a random play-out terminates with signed full-scoring returns."""
    rng = np.random.default_rng(0)
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    plies = 0
    while not state.is_terminal() and plies < MAX_RANDOM_PLIES:
        if state.is_chance_node():
            apply_sampled_chance(state, rng)
        else:
            state.apply_action(int(rng.choice(state.legal_actions())))
        plies += 1
    assert state.is_terminal()
    assert state.returns()[0] in FULL_SCORING_RETURNS
