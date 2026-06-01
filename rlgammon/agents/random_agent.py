"""A random agent for backgammon."""

import random

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.rlgammon_types import ActionGNU, ActionSetGNU


class RandomAgent(BaseAgent):
    """A random agent for backgammon."""

    def episode_setup(self) -> None:
        """A random agent needs no setup, therefore the function does nothing."""

    def choose_move(self, actions: list[int] | ActionSetGNU,
                    state: GameState) -> int | ActionGNU:  # noqa: ARG002
        """
        Choose a random move from the legal moves.

        :param actions: set of all possible actions to choose from.
        :param state: the current game state (unused by the random agent)
        :return: random action from the list of valid actions
        """
        return random.choice(list(actions))
