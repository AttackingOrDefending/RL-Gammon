"""Implementation of epsilon greedy exploration."""

import random

from rlgammon.exploration.base_exploration import BaseExploration
from rlgammon.rlgammon_types import ActionGNU, ActionSetGNU


class EpsilonGreedyExploration(BaseExploration):
    """Class implementing epsilon greedy exploration."""

    def __init__(self, start_epsilon: float = 0.95, end_epsilon: float = 0.05,
                 update_decay: float = 0.99, step_per_update: int = 5) -> None:
        """
        Initialize the epsilon-greedy exploration algorithm by setting up the start-up values.

        :param start_epsilon: the starting value of epsilon - i.e. the max chance of random action
        :param end_epsilon: the final value of epsilon - i.e. the min chance of random action
        :param update_decay: the decay of epsilon during each update
        :param step_per_update: the number of steps between each update
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.update_decay = update_decay
        self.step_per_update = step_per_update

        self.current_steps = 0
        self.current_epsilon = self.start_epsilon

    def should_explore(self) -> bool:
        """Checks whether a random value is greater than the current epsilon value
        to determine whether to explore this time step.
        If greater, than epsilon, then explore.

        :return: boolean, indicating whether to explore this time step
        """
        return random.random() > self.current_epsilon

    def explore(self, actions: list[int] | ActionSetGNU) -> int | ActionGNU:
        """
        Explore the environment by choosing a random action with a probability equal to the current value of epsilon.

        :param actions: all valid actions from the current state
        :return: the final action to execute
        """
        return random.choice(actions)

    def update(self) -> None:
        """
        Update the step counter, and if it has reached the threshold, then update the epsilon,
        by multiplying it by the decay rate.
        """
        self.current_steps += 1
        if self.current_steps % self.step_per_update == 0:
            self.current_epsilon = max(self.end_epsilon, self.current_epsilon * self.update_decay)
