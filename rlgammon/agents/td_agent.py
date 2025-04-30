"""TODO"""

import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import Action, ActionSet
from utils.utils import copy


class TDAgent(TrainableAgent):
    def __init__(self):
        """TODO."""

    def choose_move(self, actions: ActionSet, env: BackgammonEnv) -> Action:
        """
        Chooses a move to make given the current board and dice roll, which goes to the state with maximal value.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        # TODO CHECK action type
        best_action = None
        best_value = float("-inf") if self.color == WHITE else float("inf")
        for action in actions:
            env_copy = copy(env)
            state, reward, done, info = env_copy.step(action)
            if not done:
                q_values = self.model(th.tensor(state, dtype=th.float32))
                reward = q_values.item()
            if self.color == WHITE:
                if reward > best_value:
                    best_value = reward
                    best_action = action
            elif reward < best_value:
                best_value = reward
                best_action = action
        return best_action
