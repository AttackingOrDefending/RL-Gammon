"""TODO."""
import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.models.model_factory import model_factory
from rlgammon.rlgammon_types import WHITE, Action, ActionSet, State
from utils.utils import copy


class TDAgent(TrainableAgent):
    """TODO."""

    def __init__(self, model: th.nn.Module | None = None, color: int=WHITE) -> None:
        """TODO."""
        self.model = model if model else model_factory([], [])
        self.lr = 0.01
        self.gamma = 0.99
        self.lamda = 0.99
        self.color = color

    def train(self, state: State, next_state: State, reward: int, done: bool) -> float:
        """
        TODO.

        :param reward:
        :param state:
        :param next_state:
        :param done:
        :return:
        """
        p = self.model(state)
        p_next = self.model(next_state) * self.gamma

        loss = self.model.update_weights(p, reward) if done else self.model.update_weights(p, p_next)
        return loss

    def choose_move(self, actions: ActionSet, env: BackgammonEnv) -> Action:
        """
        Chooses a move to make given the current board and dice roll, which goes to the state with maximal value.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        # TODO SPLIT
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
