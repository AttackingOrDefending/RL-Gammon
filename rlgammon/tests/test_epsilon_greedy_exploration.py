"""TODO."""

from rlgammon.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration


def test_epsilon_greedy_exploration_update() -> None:
    """TODO."""
    exploration = EpsilonGreedyExploration(1, 0.05, 0.99, 5)
    # Update 1
    exploration.update()
    assert exploration.current_epsilon == 1
    # Update 2
    exploration.update()
    assert exploration.current_epsilon == 1
    # Update 3
    exploration.update()
    assert exploration.current_epsilon == 1
    # Update 4
    exploration.update()
    assert exploration.current_epsilon == 1
    # Update 5
    exploration.update()
    assert exploration.current_epsilon == 0.99  # noqa: PLR2004
