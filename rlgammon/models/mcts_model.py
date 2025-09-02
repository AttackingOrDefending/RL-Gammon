"""File implementing a model compatible with MCTS."""
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from rlgammon.models.actor_critic_model import ActorCriticModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Feature


class MCTSModel(ActorCriticModel):
    """Abstract class setting the structure for an MCTS compatible model."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """
        Construct an MCTS compatible model by creating an actor-crtic model.

        :param lr: learning rate
        :param base_layer_list: list of layers to use in the base (shared) network
        :param base_activation_list: list of activations to use in the base (shared) network
        :param policy_layer_list: list of layers to use in the policy network
        :param policy_activation_list: list of activations to use in the policy network
        :param value_layer_list: list of layers to use in the value network
        :param value_activation_list: list of activations to use in the value network
        """
        super().__init__(lr, base_layer_list, base_activation_list, policy_layer_list, policy_activation_list,
                         value_layer_list, value_activation_list)

    @abstractmethod
    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get the model value, and the chosen policy for the given state.
        A mask is used to prevent illegal actions.

        :param state: current state of the game
        :param mask: mask to prevent illegal actions
        :return: value for the state, masked policy
        """
        raise NotImplementedError
