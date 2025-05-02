"""Base class for all agents in the backgammon game."""

import random

from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import WHITE, Action, ActionSet


class BaseAgent:
    """Base class for all agents in the backgammon game."""

    def __init__(self, color: int) -> None:
        """
        TODO.

        :param color:
        """
        self.color = color

    def choose_move(self, actions: ActionSet, env: BackgammonEnv) -> Action:
        """
        Chooses a move to make given the current board and dice roll.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        raise NotImplementedError

    def roll_dice(self) -> tuple[int, int]:
        """
        Get dice rolls.

        :return: dice rolls
        """
        # TODO FIX COLOR
        if self.color == WHITE:
            return -random.randint(1, 6), -random.randint(1, 6)
        return random.randint(1, 6), random.randint(1, 6)
