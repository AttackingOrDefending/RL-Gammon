"""Test the move generation functionality of the Backgammon environment."""
import numpy as np

from rlgammon.environment import BackgammonEnv


def test_bearing_off() -> None:
    """Test the legal moves when bearing off."""
    env = BackgammonEnv()
    env.reset()
    env.backgammon.board = np.zeros(24, dtype=np.int8)
    env.backgammon.board[0] = 14
    env.backgammon.board[1] = 1
    env.backgammon.board[23] = -15
    legal_moves = env.get_legal_moves([3])
    assert legal_moves == [(3, (1, -1))]


def test_normal_moves() -> None:
    """Test the legal moves in a normal game state."""
    env = BackgammonEnv()
    env.reset()
    legal_moves = env.get_legal_moves([1, 2])
    assert legal_moves == [(1, (23, 22)), (1, (5, 4)), (1, (7, 6)), (2, (5, 3)), (2, (12, 10)), (2, (7, 5)), (2, (23, 21))]

    env = BackgammonEnv()
    env.reset()
    env.backgammon.board[0] = -2
    env.backgammon.board[1] = 2
    env.backgammon.board[2] = 1
    env.backgammon.board[3] = 1
    env.backgammon.board[4] = 3
    env.backgammon.board[5] = 5
    env.backgammon.board[6] = 1
    env.backgammon.board[11] = -2
    env.backgammon.board[15] = 1
    env.backgammon.board[17] = 1
    env.backgammon.board[19] = -3
    env.backgammon.board[20] = -1
    env.backgammon.board[22] = -3
    env.backgammon.board[23] = -2
    env.backgammon.bar[0] = 1
    env.backgammon.bar[1] = 2
    dice = [1, 1, 1, 1]
    legal_moves = env.get_legal_moves(dice)
    assert legal_moves == []


def test_bar_moves() -> None:
    """Test the legal moves when there are pieces on the bar."""
    env = BackgammonEnv()
    env.reset()
    env.backgammon.bar[0] = 1
    legal_moves = env.get_legal_moves([1, 2])
    assert legal_moves == [(1, (24, 23)), (2, (24, 22))]
