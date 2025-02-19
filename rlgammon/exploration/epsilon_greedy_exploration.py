"""Implementation of epsilon greedy exploration."""

import random

from rlgammon.exploration.base_exploration import BaseExploration


class EpsilonGreedyExploration(BaseExploration):
    """Class implementing epsilon greedy exploration."""

    def __init__(self, start_epsilon: float, end_epsilon: float, step_decay: float, step_per_update: int) -> None:
        """
        Initialize the epsilon-greedy exploration algorithm by setting up the start-up values.

        :param start_epsilon: the starting value of epsilon - i.e. the max chance of random action
        :param end_epsilon: the final value of epsilon - i.e. the min chance of random action
        :param step_decay: the decay of epsilon during each update
        :param step_per_update: the number of steps between each update
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.step_decay = step_decay
        self.step_per_update = step_per_update

        self.current_steps = 0
        self.current_epsilon = self.start_epsilon

    def explore(self, action: int, valid_actions: list[int]) -> int:
        """
        Explore the environment by choosing a random action with a probability equal to the current value of epsilon.

        :param action: current action chosen by the agent
        :param valid_actions: all valid actions from the current state
        :return: the final action to execute
        """
        if random.random() > self.current_epsilon:
            action = random.choice(valid_actions)
        return action

    def update(self) -> None:
        """
        Update the step counter, and if it has reached the threshold.
        Then update the epsilon, by multiplying it by the decay rate.
        """
        self.current_steps += 1
        if self.current_steps % self.step_per_update == 0:
            self.current_epsilon = max(self.end_epsilon, self.current_epsilon * self.step_decay)
