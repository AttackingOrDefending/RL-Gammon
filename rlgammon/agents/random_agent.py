"""A random agent for backgammon."""

import random

import pyspiel  # type: ignore[import-not-found]

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment import BackgammonEnv  # type: ignore[attr-defined]
from rlgammon.rlgammon_types import ActionInfoTuple, ActionSetGNU


class RandomAgent(BaseAgent):
    """A random agent for backgammon."""

    def episode_setup(self) -> None:
        """A random agent needs no setup, therefore the function does nothing."""

    def choose_move(self, actions: list[int] | ActionSetGNU,
                    state: pyspiel.BackgammonState | BackgammonEnv) -> ActionInfoTuple: # noqa: ARG002
        """
        Choose a random move from the legal moves.

        :param actions: set of all possible actions to choose from.
        :param state: the current state of the game or environment with the current state if GNU
        :return: random action from the list of valid actions
        """
        return random.choice(list(actions)), None
