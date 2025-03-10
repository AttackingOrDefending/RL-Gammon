"""Test the get_winner functionality of the Backgammon environment."""

from rlgammon.environment import BackgammonEnv


def test_get_winner() -> None:
    """Test the get_winner method of the Backgammon environment."""

    env = BackgammonEnv()
    env.reset()

    env.backgammon.board[:] = 0
    env.backgammon.board[23] = -15
    env.backgammon.off[0] = 15
    assert env.backgammon.get_winner() == 3
