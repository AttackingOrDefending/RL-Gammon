"""Base class for all agents in the backgammon game."""

from abc import abstractmethod
import random

from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import BLACK, MAX_DICE, MIN_DICE, WHITE, Action, ActionSet


class BaseAgent:
    """Base class for all agents in the backgammon game."""

    def __init__(self, color: int) -> None:
        """
        Construct the base agent by assigning it to a player.

        :param color: 0 or 1 representing which player the agent controls
        """
        self.color = color

    @abstractmethod
    def episode_setup(self) -> None:
        """Prepare the agent for a start of an episode."""
        raise NotImplementedError

    def choose_move(self, actions: ActionSet, env: BackgammonEnv) -> Action:
        """
        Chooses a move to make given the current board and dice roll.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        raise NotImplementedError

    def flip_color(self) -> None:
        """Flip the color of the agent, i.e. make it control the opposite player."""
        self.color = WHITE if self.color == BLACK else BLACK

    def set_color(self, color: int) -> None:
        """
        Set a new color of the agent to indicate that it should control a different player.

        :param color: new color of the agent, i.e. new player it should control
        """
        self.color = color

    def roll_dice(self) -> tuple[int, int]:
        """
        Get dice rolls.

        :return: dice rolls
        """
        if self.color == WHITE:
            return -random.randint(MIN_DICE, MAX_DICE), -random.randint(MIN_DICE, MAX_DICE)
        return random.randint(MIN_DICE,  MAX_DICE), random.randint(MIN_DICE, MAX_DICE)
