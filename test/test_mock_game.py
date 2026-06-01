"""Test the pure-Python mock game used for engine-free tests."""

from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.game.feature_extractor import N_OBS
from rlgammon.game.mock_game import (
    CHANCE_DISTRIBUTION,
    CHANCE_PLAYER,
    MockGame,
)
from rlgammon.rlgammon_types import BLACK, WHITE

WIN_RETURN = 1.0
LOSS_RETURN = -1.0
PROBABILITY_TOLERANCE = 1e-9


def _is_winning_action(state: GameState, action: int) -> bool:
    """
    Return whether applying the action on a clone reaches a terminal state.

    :param state: the state to test the action from
    :param action: the action to apply
    :return: true if the action ends the game
    """
    child = state.clone()
    child.apply_action(action)
    return child.is_terminal()


def test_mock_game_satisfies_protocols() -> None:
    """Test that the mock game and its states satisfy the runtime-checkable protocols."""
    game = MockGame()
    assert isinstance(game, BackgammonGame)
    assert isinstance(game.new_initial_state(), GameState)


def test_initial_state_is_chance_node() -> None:
    """Test that a fresh state is a non-terminal chance node for the opening roll."""
    state = MockGame().new_initial_state()
    assert state.is_chance_node()
    assert not state.is_terminal()
    assert state.current_player() == CHANCE_PLAYER


def test_chance_distribution_is_valid() -> None:
    """Test that the dice distribution matches the constant and is a valid probability distribution."""
    state = MockGame().new_initial_state()
    outcomes = state.chance_outcomes()
    assert outcomes == CHANCE_DISTRIBUTION
    assert abs(sum(prob for _, prob in outcomes) - 1.0) < PROBABILITY_TOLERANCE


def test_observation_tensor_length() -> None:
    """Test that the observation tensor has the expected full length."""
    state = MockGame().new_initial_state()
    assert len(state.observation_tensor(WHITE)) == N_OBS


def test_contrived_win_in_one_has_single_winning_action() -> None:
    """Test that exactly one legal action wins immediately from the contrived state."""
    state = MockGame.contrived_win_in_one(WHITE)
    winning = [action for action in state.legal_actions() if _is_winning_action(state, action)]
    assert winning == [MockGame.winning_action()]


def test_contrived_win_returns_signed_points() -> None:
    """Test that playing the winning action yields the expected signed returns."""
    state = MockGame.contrived_win_in_one(WHITE)
    state.apply_action(MockGame.winning_action())
    assert state.is_terminal()
    assert state.returns()[WHITE] == WIN_RETURN
    assert state.returns()[BLACK] == LOSS_RETURN


def test_clone_is_independent() -> None:
    """Test that mutating a clone does not affect the original state."""
    state = MockGame.contrived_win_in_one(WHITE)
    clone = state.clone()
    clone.apply_action(MockGame.winning_action())
    assert clone.is_terminal()
    assert not state.is_terminal()


def test_contrived_win_for_black() -> None:
    """Test that the contrived winning position also works for the black player."""
    state = MockGame.contrived_win_in_one(BLACK)
    assert state.current_player() == BLACK
    state.apply_action(MockGame.winning_action())
    assert state.returns()[BLACK] == WIN_RETURN
