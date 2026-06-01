"""Tests for the deadline-driven anytime behaviour of star-minimax and stochastic MCTS."""

import time

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.mcts import StochasticMCTS
from rlgammon.rlgammon_types import WHITE

# A short slack (seconds) added to "now" to give the search a real, but tiny, time budget.
DEADLINE_SLACK = 0.2
# A fixed seed so the MCTS checks are deterministic.
SEED = 12345


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


def test_star_minimax_deadline_returns_winning_action() -> None:
    """Test that star-minimax with a live deadline still finds the immediately-winning action."""
    state = MockGame.contrived_win_in_one(WHITE)
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=3)
    result = search.search(state, deadline=time.monotonic() + DEADLINE_SLACK)
    assert result.best_action == MockGame.winning_action()
    assert result.best_action in state.legal_actions()


def test_star_minimax_passed_deadline_is_graceful() -> None:
    """Test that an already-passed deadline still yields a legal action rather than failing."""
    state = MockGame.contrived_win_in_one(WHITE)
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=3)
    result = search.search(state, deadline=time.monotonic() - DEADLINE_SLACK)
    assert result.best_action in state.legal_actions()


def test_mcts_deadline_returns_valid_result() -> None:
    """Test that MCTS with a live deadline returns a legal action and runs at least one simulation."""
    state = MockGame.contrived_win_in_one(WHITE)
    rng = np.random.default_rng(SEED)
    search = StochasticMCTS(ConstantEvaluator(0.0), max_depth=2, rng=rng)
    result = search.search(state, deadline=time.monotonic() + DEADLINE_SLACK)
    assert result.best_action in state.legal_actions()
    assert sum(search.get_visit_counts().values()) >= 1


def test_mcts_passed_deadline_runs_one_simulation() -> None:
    """Test that an already-passed deadline still runs a single simulation (the minimum guarantee)."""
    state = MockGame.contrived_win_in_one(WHITE)
    rng = np.random.default_rng(SEED)
    search = StochasticMCTS(ConstantEvaluator(0.0), max_depth=2, rng=rng)
    result = search.search(state, deadline=time.monotonic() - DEADLINE_SLACK)
    assert result.best_action in state.legal_actions()
    assert sum(search.get_visit_counts().values()) >= 1
