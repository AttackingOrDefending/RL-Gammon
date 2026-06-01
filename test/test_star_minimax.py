"""Tests for the star-minimax expectiminimax search on the pyspiel-free mock game."""

import pytest

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.planning_errors.planning_errors import SearchDepthError
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


def brute_force_expectimax(state: GameState, depth: int, mover: int, evaluator: ConstantEvaluator) -> float:
    """
    Compute the plain (unpruned) negamax expectimax value of a state, as a reference.

    :param state: the game state to evaluate
    :param depth: the remaining search depth in decision plies
    :param mover: the player to move at a decision node (perspective of the returned value)
    :param evaluator: the leaf evaluator for non-terminal frontier states
    :return: ``mover``'s expectimax value of the state
    """
    if state.is_terminal():
        return state.returns()[mover]
    if state.is_chance_node():
        expectation = 0.0
        for outcome, prob in state.chance_outcomes():
            child = state.clone()
            child.apply_action(outcome)
            opponent = child.current_player()
            expectation += prob * -brute_force_expectimax(child, depth - 1, opponent, evaluator)
        return expectation
    if depth <= 0:
        return evaluator.evaluate(state, mover)
    best = -float("inf")
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        best = max(best, brute_force_expectimax(child, depth, mover, evaluator))
    return best


def test_picks_winning_action() -> None:
    """Test that star-minimax returns the unique immediately-winning action."""
    state = MockGame.contrived_win_in_one(WHITE)
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=1)
    result = search.search(state)
    assert result.best_action == MockGame.winning_action()


def test_picks_winning_action_deeper() -> None:
    """Test that star-minimax still returns the winning action at a larger depth."""
    state = MockGame.contrived_win_in_one(WHITE)
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=3)
    result = search.search(state)
    assert result.best_action == MockGame.winning_action()


def test_value_matches_brute_force() -> None:
    """Test that the star-minimax root value equals the plain expectimax reference value."""
    state = MockGame.contrived_win_in_one(WHITE)
    depth = 3
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=depth)
    result = search.search(state)
    reference = brute_force_expectimax(state.clone(), depth, WHITE, ConstantEvaluator(0.0))
    assert result.value == pytest.approx(reference)


def test_value_matches_brute_force_without_star2() -> None:
    """Test value-preservation of the star1-only variant against the plain expectimax reference."""
    state = MockGame.contrived_win_in_one(WHITE)
    depth = 3
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=depth, use_star2=False)
    result = search.search(state)
    reference = brute_force_expectimax(state.clone(), depth, WHITE, ConstantEvaluator(0.0))
    assert result.value == pytest.approx(reference)


def test_winning_value_is_positive() -> None:
    """Test that the winning root has a strictly positive value for the mover."""
    state = MockGame.contrived_win_in_one(WHITE)
    search = StarMinimax(ConstantEvaluator(0.0), max_depth=1)
    result = search.search(state)
    assert result.value > 0.0


def test_search_depth_error() -> None:
    """Test that constructing a search with a depth below 1 raises SearchDepthError."""
    with pytest.raises(SearchDepthError):
        StarMinimax(ConstantEvaluator(0.0), max_depth=0)
