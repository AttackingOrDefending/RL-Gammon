"""Test no exploration."""

import pytest

from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.exploration.no_exploration import NoExploration


def test_get_epsilon_from_string() -> None:
    """Test conversion of the string representation of epsilon-greedy exploration to enum."""
    assert PossibleExploration.get_enum_from_string("NO") == PossibleExploration.NO_EXPLORATION


def test_should_not_explore() -> None:
    """Test that the no exploration class doesn't request for exploration."""
    exploration = NoExploration()
    assert not exploration.should_explore()
    assert not exploration.should_explore()


def test_exploration_not_allowed() -> None:
    """
    Test that an exception is raised when trying to explore with the no exploration class,
    as that's not allowed.
    """
    exploration = NoExploration()
    with pytest.raises(NotImplementedError) as excinfo:
        exploration.explore([])
    assert excinfo.type is NotImplementedError
