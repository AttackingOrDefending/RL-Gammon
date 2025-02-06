from rlgammon.agents.base_agent import BaseAgent
import random


class RandomAgent(BaseAgent):
    def choose_move(self, board, dice):
        actions = board.get_legal_moves(dice)
        if not actions:
            return None, None
        return random.choice(actions)
