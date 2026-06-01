"""Tests for the stochastic UCT MCTS on the pyspiel-free mock game."""

import numpy as np
import pytest

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import CHANCE_DISTRIBUTION, MockGame
from rlgammon.planning.mcts import StochasticMCTS
from rlgammon.planning.planning_errors.planning_errors import SearchDepthError
from rlgammon.rlgammon_types import WHITE

# A fixed seed so the statistical checks are deterministic.
SEED = 12345
# Number of samples for the chance-distribution check.
NUM_CHANCE_SAMPLES = 20000
# Absolute tolerance for the empirical-vs-true probability comparison.
PROB_TOLERANCE = 0.02


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


def test_picks_winning_action() -> None:
    """Test that MCTS returns the unique immediately-winning action with enough simulations."""
    state = MockGame.contrived_win_in_one(WHITE)
    rng = np.random.default_rng(SEED)
    search = StochasticMCTS(ConstantEvaluator(0.0), max_depth=2, num_simulations=400, rng=rng)
    result = search.search(state)
    assert result.best_action == MockGame.winning_action()


def test_winning_action_most_visited() -> None:
    """Test that the winning action accrues the most visits at the root."""
    state = MockGame.contrived_win_in_one(WHITE)
    rng = np.random.default_rng(SEED)
    search = StochasticMCTS(ConstantEvaluator(0.0), max_depth=2, num_simulations=400, rng=rng)
    search.search(state)
    counts = search.get_visit_counts()
    winning = MockGame.winning_action()
    assert counts[winning] == max(counts.values())
    assert all(counts[winning] >= counts[action] for action in counts)


def test_chance_sampling_matches_distribution() -> None:
    """Test that seeded chance sampling approximately matches the mock's dice distribution."""
    rng = np.random.default_rng(SEED)
    search = StochasticMCTS(ConstantEvaluator(0.0), max_depth=1, rng=rng)
    chance_state = MockGame().new_initial_state()
    assert chance_state.is_chance_node()

    tallies: dict[int, int] = {action: 0 for action, _ in CHANCE_DISTRIBUTION}
    for _ in range(NUM_CHANCE_SAMPLES):
        tallies[search._sample_chance(chance_state)] += 1

    for action, prob in CHANCE_DISTRIBUTION:
        empirical = tallies[action] / NUM_CHANCE_SAMPLES
        assert empirical == pytest.approx(prob, abs=PROB_TOLERANCE)


def test_search_depth_error() -> None:
    """Test that constructing MCTS with a depth below 1 raises SearchDepthError."""
    with pytest.raises(SearchDepthError):
        StochasticMCTS(ConstantEvaluator(0.0), max_depth=0)
