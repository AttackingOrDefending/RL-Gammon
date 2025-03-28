"""A random agent for backgammon."""

import random

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class RandomAgent(BaseAgent):
    """A random agent for backgammon."""

    def choose_move(self, board: BackgammonEnv) -> tuple[int, MovePart]:
        """
        Choose a random move from the legal moves.

        :param board: current state of the game
        :return: random action from the list of valid actions
        """
        valid_actions = board.get_all_complete_moves()
        return random.choice([move for _, move in valid_actions]) if valid_actions else []

    def choose_move_deprecated(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """Choose a random move from the legal moves."""
        board_copy = board.copy()
        dice = dice.copy()
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
