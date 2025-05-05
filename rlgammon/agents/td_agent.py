"""File implementing an agent trained with td-learning."""
import pathlib
from uuid import UUID

import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.models.td_model import TDModel
from rlgammon.rlgammon_types import WHITE, Action, ActionSet, State
from utils.utils import copy


class TDAgent(TrainableAgent):
    """Class implementing an agent trained with td."""

    def __init__(self, pre_made_model_file_name: str | None = None, lr: float = 0.01,
                 gamma: float = 0.99, lamda: float = 0.99, seed: int = 123, color: int=WHITE,
                 layer_list: LayerList = None, activation_list: ActivationList = None) -> None:
        """
        Construct a td-agent by first loading a model or creating a new one with the given layers and activations,
        and the initializing various parameters used in td learning.

        :param pre_made_model_file_name: file name of a previously trained model, None if a new model is to be used
        :param lr: learning rate
        :param gamma: future reward discount
        :param lamda: trace decay parameters (how much to value distant states)
        :param seed: seed for random number generator of torch and the python random package
        :param color: 0 or 1 representing which player the agent controls
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        """
        super().__init__(color)
        self.model = self.load(pre_made_model_file_name) \
            if pre_made_model_file_name else TDModel(lr, gamma, lamda, seed, layer_list, activation_list)
        self.gamma = gamma

    def episode_setup(self) -> None:
        """Prepare the agent for a training episode by initializing the model's eligibility traces."""
        self.model.init_eligibility_traces()

    def train(self, state: State, next_state: State, reward: int, done: bool) -> float:
        """
        Update the weights of the model according to the td algorithm.
        If the state is terminal use reward for next state value, else use the model estimation.

        :param state: the current state
        :param next_state: the next state
        :param reward: the reward from the environment associated with the current state
        :param done: whether the state is the last one of the round
        :return:
        """
        p = self.model(state)
        p_next = self.model(next_state) # * self.gamma
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

    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str | None = None) -> None:
        """
        Save the td model.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        :param main_filename: name of the file under which the agent is to be saved
        """
        agent_main_filename = f"{main_filename}-{training_session_id}-({session_save_count}).pt"
        agent_file_path = pathlib.Path(__file__).parent
        agent_file_path = agent_file_path.joinpath("saved_agents/")
        agent_file_path.mkdir(parents=True, exist_ok=True)
        th.save(self.model.state_dict(), agent_file_path.joinpath(agent_main_filename))

    def load(self, agent_main_filename: str) -> th.nn.Module:
        """
        Load the td model.

        :param agent_main_filename: name of the file under which the agent is saved
        :return: the loaded agent model
        """
        agent_file_path = pathlib.Path(__file__).parent
        agent_file_path = agent_file_path.joinpath("saved_agents/")
        return th.load(agent_file_path.joinpath(agent_main_filename), weights_only=False)
