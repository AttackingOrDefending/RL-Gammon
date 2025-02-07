import numpy as np
from rlgammon.environment import text_renderer


class Backgammon:
    def __init__(self):
        self.board = np.zeros(24, dtype=np.int8)
        self.bar = np.zeros(2, dtype=np.int8)
        self.off = np.zeros(2, dtype=np.int8)
        self.reset()

    def reset(self):
        # Minus is black, plus is white
        self.board = np.zeros(24, dtype=np.int8)
        self.bar = np.zeros(2, dtype=np.int8)
        self.off = np.zeros(2, dtype=np.int8)
        self.board[0] = -2
        self.board[5] = 5
        self.board[7] = 3
        self.board[11] = -5
        self.board[12] = 5
        self.board[16] = -3
        self.board[18] = -5
        self.board[23] = 2

    def flip(self):
        self.board = -self.board
        self.board = np.flipud(self.board)
        self.bar = np.flipud(self.bar)
        self.off = np.flipud(self.off)

    def get_legal_moves(self, dice):
        possible_moves = {roll: set() for roll in set(dice)}

        # There are men on the bar.
        if self.bar[0] > 0:
            for roll in dice:
                if self.board[24 - roll] >= -1:
                    possible_moves[roll].add((24, 24 - roll))
            return possible_moves

        # Normal moves.
        for roll in dice:
            for loc in np.argwhere(self.board > 0).reshape(-1):
                if loc - roll < 0:
                    continue
                if self.board[loc - roll] >= -1:
                    possible_moves[roll].add((loc, loc - roll))

        # Bear off.
        if np.all(self.board[6:] <= 0):
            for roll in dice:
                for loc in np.argwhere(self.board > 0).reshape(-1):
                    if loc - roll < 0:
                        possible_moves[roll].add((loc, -1))
        return possible_moves

    def make_move(self, move):
        captures = 0
        beared_off = 0
        if move[1] == -1:
            self.board[move[0]] -= 1
            beared_off = 1
            self.off[0] += 1
        else:
            if move[0] == 24:
                self.bar[0] -= 1
            else:
                self.board[move[0]] -= 1
            if self.board[move[1]] == -1:
                self.board[move[1]] = 1
                self.bar[1] += 1
                captures = 1
            else:
                self.board[move[1]] += 1
        return captures, beared_off

    def is_terminal(self):
        return np.all(self.board <= 0) or np.all(self.board >= 0)

    def get_winner(self):
        if np.all(self.board <= 0):
            return 1
        return 0

    def render(self):
        return text_renderer.text_render(self)

    def copy(self):
        bg = Backgammon()
        bg.board = self.board.copy()
        bg.bar = self.bar.copy()
        bg.off = self.off.copy()
        return bg


if __name__ == "__main__":
    bg = Backgammon()
    bg.board[3] = 8
    bg.bar[0] = 9
    bg.bar[1] = 3
    bg.off[0] = 5
    bg.off[1] = 8
    print(bg.render())
