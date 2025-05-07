#type: ignore  # noqa: PGH003

"""File implementing an agent trained with td-learning."""
import pathlib
from uuid import UUID

import pyspiel  # type: ignore[import-not-found]
import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment import BackgammonEnv  # type: ignore[attr-defined]
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.models.td_model import TDModel
from rlgammon.rlgammon_types import WHITE, ActionGNU, ActionSetGNU, Features
from utils.utils import copy


class TDAgent(TrainableAgent):
    """Class implementing an agent trained with td."""

    def __init__(self, pre_made_model_file_name: str | None = None, lr: float = 0.01,
                 gamma: float = 0.99, lamda: float = 0.99, seed: int = 123, color: int = WHITE,
                 layer_list: LayerList = None, activation_list: ActivationList = None, dtype: str = "float32") -> None:
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
        :param dtype: the data type of the model
        """
        super().__init__(color)
        self.model = None
        if pre_made_model_file_name:
            self.model = self.load(pre_made_model_file_name)
        elif layer_list and activation_list:
            self.model = TDModel(lr, gamma, lamda, layer_list, activation_list, seed, dtype)
        self.gamma = gamma

    def episode_setup(self) -> None:
        """Prepare the agent for a training episode by initializing the model's eligibility traces."""
        self.model.init_eligibility_traces()

    def evaluate_position(self, state: Features, decay: bool = False) -> th.Tensor:
        """
        Evaluate the given position using the agent model.

        :param state: state to evaluate
        :param decay: flag whether to decay the value or not
        :return: th tensor storing the value of the provided state
        """
        return self.model(state) * self.gamma if decay else self.model(state)

    def train(self, p: th.Tensor, p_next: th.Tensor) -> float:
        """
        Update the weights of the model according to the td algorithm.
        If the state is terminal use reward for next state value, else use the model estimation.

        :param p: value of current state
        :param p_next: value of the next state or final reward if terminal state
        :return: loss associated with update
        """
        return self.model.update_weights(p, p_next)

    def choose_move(self, actions: list[int] | ActionSetGNU,
                    state: pyspiel.BackgammonState | BackgammonEnv) -> int | ActionGNU:
        """
        Chooses a move to make given the current board and dice roll, which goes to the state with maximal value.

        :param actions: set of all possible actions to choose from.
        :param state: the current state of the game.
        :return: the chosen move to make.
        """
        best_action = None
        best_value = -10000000 if self.color == WHITE else 10000000
        for action in actions:
            state_copy = copy(state)
            state_copy.apply_action(action)
            if state.is_chance_node():
                # Always roll the dice, so that the side to move is included in the input.
                state_copy.apply_action(7)
            if not state_copy.is_terminal():
                # Remove the last 2 elements, which are the dice. Always from white perspective.
                features = state_copy.observation_tensor(WHITE)[:198]
                q_values = self.model(th.tensor(features, dtype=th.float32))
                reward = q_values.item()
            else:
                # If terminal state, use the actual reward (negative is black wins).
                reward = state_copy.returns()[WHITE]
            if self.color == WHITE:
                if reward > best_value:
                    best_value = reward
                    best_action = action
            elif reward < best_value:
                best_value = reward
                best_action = action
        return best_action

    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str = "td-backgammon") -> None:
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

    def get_model(self) -> th.nn.Module:
        """
        Get the model this agent is using.

        :return: the agent model if it has one
        """
        return self.model
