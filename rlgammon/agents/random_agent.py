"""A random agent for backgammon."""

import random

from rlgammon.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """A random agent for backgammon."""

    def episode_setup(self) -> None:
        """A random agent needs no setup, therefore the function does nothing."""

    def choose_move(self, actions: list[int], state) -> int:
        """
        Choose a random move from the legal moves.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: random action from the list of valid actions
        """
        return random.choice(list(actions)) if actions else None
