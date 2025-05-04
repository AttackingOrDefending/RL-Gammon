"""TODO."""
import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.models.td_model import TDModel
from rlgammon.rlgammon_types import WHITE, Action, ActionSet, State
from utils.utils import copy


class TDAgent(TrainableAgent):
    """TODO."""

    def __init__(self, pre_made_model: th.nn.Module | None = None, lr: float = 0.01,
                 gamma: float = 0.99, lamda: float = 0.99, seed: int = 123, color: int=WHITE,
                 layer_list: LayerList = None, activation_list: ActivationList = None) -> None:
        """
        TODO.

        :param pre_made_model:
        :param lr:
        :param gamma:
        :param lamda:
        :param color:
        """
        super().__init__(color)
        self.model = pre_made_model if pre_made_model else TDModel(lr, gamma, lamda, seed, layer_list, activation_list)

        self.gamma = gamma

    def episode_setup(self) -> None:
        """TODO."""
        self.model.init_eligibility_traces()

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

        return self.model.update_weights(p, reward) if done else self.model.update_weights(p, p_next)

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
