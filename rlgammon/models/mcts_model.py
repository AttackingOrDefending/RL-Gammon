"""TODO."""
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from rlgammon.models.actor_critic_model import ActorCriticModel
from rlgammon.models.model_types import ActivationList, ActorCriticOutput, LayerList
from rlgammon.rlgammon_types import Feature


class MCTSModel(ActorCriticModel):
    """TODO."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """TODO."""
        super().__init__(lr, base_layer_list, base_activation_list, policy_layer_list, policy_activation_list,
                         value_layer_list, value_activation_list)

    @abstractmethod
    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """TODO."""
