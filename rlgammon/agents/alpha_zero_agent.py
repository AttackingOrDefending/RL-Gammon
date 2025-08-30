"""TODO."""
import pathlib
from typing import Any
from uuid import UUID

import numpy as np
from numpy._typing import NDArray
from open_spiel.python.algorithms import mcts
import pyspiel
import torch as th

from rlgammon.agents.agent_errors.agent_errors import AlphaZeroNotSetupError
from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.evaluators.alpha_zero_evaluator import AlphaZeroEvaluator
from rlgammon.models.alpha_zero_model import AlphaZeroModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import WHITE, ActionGNU, ActionSetGNU, Feature


class AlphaZeroAgent(TrainableAgent):
    """TODO."""

    def __init__(self, mcts_evaluator: AlphaZeroEvaluator, game: pyspiel.Game, uct_c: float,
                 max_simulations: int, temperature: float, pre_made_model_file_name: str | None = None,
                 lr: float = 0.01, gamma: float = 0.99, color: int = WHITE,
                 base_layer_list: LayerList = None, base_activation_list: ActivationList = None,
                 policy_layer_list: LayerList = None, policy_activation_list: ActivationList = None,
                 value_layer_list: LayerList = None, value_activation_list: ActivationList = None) -> None:
        """TODO."""
        super().__init__(color)
        self.mcts_evaluator = mcts_evaluator
        self.max_simulations = max_simulations
        self.uct_c = uct_c
        self.game = game
        self.temperature = temperature
        self.mcts_bot = None
        self.setup = False
        self.gamma = gamma
        self.model = self.load(pre_made_model_file_name) \
            if pre_made_model_file_name else AlphaZeroModel(lr, base_layer_list, base_activation_list,
                                                            policy_layer_list, policy_activation_list,
                                                            value_layer_list, value_activation_list)

    def evaluate_position(self, state: Feature, decay: bool = False) -> th.Tensor:
        """TODO."""
        value, policy = self.model(state)
        return value * self.gamma if decay else value, policy

    def position_policy(self, state: Feature) -> th.Tensor:
        """TODO."""
        _, policy = self.model(state)
        return policy

    def train(self, action_info: Any, reward: int, state: Feature, next_state: Feature, _: bool) -> float:
        """
        TODO.

        :param action_info:
        :param reward:
        :param state:
        :param next_state:
        :param _:
        :return:
        """
        mcts_probs = th.tensor(action_info)
        reward = th.tensor(reward, dtype=th.float32)
        return self.model.update_weights(mcts_probs, reward, state, next_state)

    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str = "alpha-zero-backgammon") -> None:
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
        return th.load(agent_file_path.joinpath(agent_main_filename), weights_only=False)  # type: ignore[no-any-return]

    def get_model(self) -> th.nn.Module | None:
        """"
        Get the model this agent is using.

        :return: the agent model if it has one
        """
        return self.model

    def episode_setup(self) -> None:
        """TODO."""
        self.mcts_bot = mcts.MCTSBot(self.game, uct_c=self.uct_c, max_simulations=self.max_simulations,
                                     evaluator=self.mcts_evaluator, dont_return_chance_node=True)
        self.setup = True

    def choose_move(self, actions: list[int] | ActionSetGNU,
                    state: pyspiel.BackgammonState | BackgammonEnv) -> tuple[int | ActionGNU, Any]:
        """TODO."""
        if not self.setup:
            raise AlphaZeroNotSetupError

        root = self.mcts_bot.mcts_search(state)  # type: ignore[attr-defined]
        policy = np.zeros(self.game.num_distinct_actions())
        for c in root.children:
            policy[c.action] = c.explore_count

        match self.temperature:
            case 0:
                policy /= policy.sum()
                return np.argmax(policy), policy
            case float("inf"):
                policy /= policy.sum()
                return np.random.choice(self.game.num_distinct_actions()), policy
            case _:
                policy = policy ** (1 / self.temperature)
                policy /= policy.sum()
                return np.random.choice(self.game.num_distinct_actions(), p=policy), policy
