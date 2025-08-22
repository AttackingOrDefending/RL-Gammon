#type: ignore  # noqa: PGH003

"""File implementing an agent trained with td-learning."""
import pathlib
from uuid import UUID

import pyspiel  # type: ignore[import-not-found]
import torch as th

from rlgammon.agents.agent_errors.agent_errors import EligibilityTracesNotInitializedError
from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment import BackgammonEnv  # type: ignore[attr-defined]
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.models.td_model import TDModel
from rlgammon.rlgammon_types import INF, NEG_INF, WHITE, ActionInfoTuple, ActionSetGNU, Feature
from utils.utils import copy


class TDAgent(TrainableAgent):
    """Class implementing an agent trained with td."""

    def __init__(self, pre_made_model_file_name: str | None = None, lr: float = 0.01,
                 gamma: float = 0.99, lamda: float = 0.99, color: int = WHITE,
                 layer_list: LayerList = None, activation_list: ActivationList = None) -> None:
        """
        Construct a td-agent by first loading a model or creating a new one with the given layers and activations,
        and the initializing various parameters used in td learning.

        :param pre_made_model_file_name: file name of a previously trained model, None if a new model is to be used
        :param lr: learning rate
        :param gamma: future reward discount
        :param lamda: trace decay parameters (how much to value distant states)
        :param color: 0 or 1 representing which player the agent controls
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        """
        super().__init__(color)
        self.model =  self.load(pre_made_model_file_name) \
            if pre_made_model_file_name else TDModel(lr, gamma, lamda, layer_list, activation_list)
        self.setup = False
        self.gamma = gamma

    def episode_setup(self) -> None:
        """Prepare the agent for a training episode by initializing the model's eligibility traces."""
        self.setup = True
        self.model.init_eligibility_traces()

    def evaluate_position(self, state: Feature, decay: bool = False) -> th.Tensor:
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
        # Raise an error if training is attempted without prior initialization of eligibility traces
        if not self.setup:
            raise EligibilityTracesNotInitializedError

        return self.model.update_weights(p, p_next)

    def choose_move(self, actions: list[int] | ActionSetGNU,
                    state: pyspiel.BackgammonState | BackgammonEnv) -> ActionInfoTuple:
        """
        Chooses a move to make given the current board and dice roll, which goes to the state with maximal value.

        :param actions: set of all possible actions to choose from.
        :param state: the current state of the game.
        :return: the chosen move to make. No action info
        """
        best_action = None
        color = state.current_player()
        best_value = NEG_INF if color == WHITE else INF
        for action in actions:
            state_copy = copy(state)
            state_copy.apply_action(action)
            if state.is_chance_node():
                # Always roll the dice, so that the side to move is included in the input.
                state_copy.apply_action(7)
            if not state_copy.is_terminal():
                # Remove the last 2 elements, which are the dice. Always from white perspective.
                features = state_copy.observation_tensor(WHITE)[:198]
                q_values = self.model(features)
                reward = q_values.item()
            else:
                # If terminal state, use the actual reward (negative is black wins).
                reward = state_copy.returns()[WHITE]
            if color == WHITE:
                if reward > best_value:
                    best_value = reward
                    best_action = action
            elif reward < best_value:
                best_value = reward
                best_action = action
        return best_action, None

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
