"""A random agent for backgammon."""

import random

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class RandomAgent(BaseAgent):
    """A random agent for backgammon."""

    def choose_move(self, actions: set, env: BackgammonEnv) -> tuple[int, MovePart] | None:  # noqa: ARG002
        """
        Choose a random move from the legal moves.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: random action from the list of valid actions
        """
        return random.choice(list(actions))
