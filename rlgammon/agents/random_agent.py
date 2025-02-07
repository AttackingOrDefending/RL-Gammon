from rlgammon.agents.base_agent import BaseAgent
import random


class RandomAgent(BaseAgent):
    def choose_move(self, board, dice):
        board_copy = board.copy()
        chosen_actions = []
        while dice:
            actions = board_copy.get_legal_moves(dice)
            if not actions:
                break
            roll, action = random.choice(actions)
            chosen_actions.append((roll, action))
            dice.remove(roll)
            board_copy.backgammon.make_move(action)
        return chosen_actions
