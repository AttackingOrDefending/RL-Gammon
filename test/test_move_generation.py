from rlgammon.environment import BackgammonEnv
import numpy as np


def test_bearing_off():
    env = BackgammonEnv()
    env.reset()
    env.backgammon.board = np.zeros(24, dtype=np.int8)
    env.backgammon.board[0] = 14
    env.backgammon.board[1] = 1
    env.backgammon.board[23] = -15
    legal_moves = env.get_legal_moves([3])
    assert legal_moves == [(3, (1, -1))]


def test_normal_moves():
    env = BackgammonEnv()
    env.reset()
    legal_moves = env.get_legal_moves([1, 2])
    assert legal_moves == [(1, (23, 22)), (1, (5, 4)), (1, (7, 6)), (2, (5, 3)), (2, (12, 10)), (2, (7, 5)), (2, (23, 21))]


def test_bar_moves():
    env = BackgammonEnv()
    env.reset()
    env.backgammon.bar[0] = 1
    legal_moves = env.get_legal_moves([1, 2])
    assert legal_moves == [(1, (24, 23)), (2, (24, 22))]
