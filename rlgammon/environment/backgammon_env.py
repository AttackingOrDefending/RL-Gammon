import gymnasium as gym
from rlgammon.environment import backgammon as bg
from rlgammon.environment import renderer
import numpy as np
import random


class BackgammonEnv(gym.Env):
    def __init__(self):
        super(BackgammonEnv, self).__init__()
        self.backgammon = bg.Backgammon()
        self.max_moves = 500
        self.moves = 0

    def get_input(self):
        our_pieces = self.backgammon.board.copy()
        our_pieces[our_pieces < 0] = 0
        enemy_pieces = -self.backgammon.board.copy()
        enemy_pieces[enemy_pieces < 0] = 0
        return np.concatenate([our_pieces, enemy_pieces, self.backgammon.bar, self.backgammon.off])

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.backgammon.reset()
        self.moves = 0
        return self.get_input(), {}

    def roll_dice(self):
        rolls = [random.randint(1, 6), random.randint(1, 6)]
        if rolls[0] == rolls[1]:
            return rolls * 2
        return rolls

    def flip(self):
        self.backgammon.flip()
        return self.get_input()

    def step(self, action):
        self.moves += 1
        self.backgammon.make_move(action)
        done = self.backgammon.is_terminal()
        reward = 0
        if done:
            if self.backgammon.get_winner() == 1:
                reward = 1
            else:
                reward = -1
        return self.get_input(), reward, done, self.moves >= self.max_moves and not done, {}

    def render(self, mode='human'):
        if mode == 'human':
            render = renderer.BackgammonRenderer()
            render.render(self.backgammon.board, self.backgammon.bar, self.backgammon.off)
        else:
            print(self.backgammon.render_backgammon_board_aligned())

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_legal_moves(self, dice):
        return self.backgammon.get_legal_moves(dice)
