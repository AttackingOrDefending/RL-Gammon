"""TODO."""
import torch as th

from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_types import ActivationList, ActorCriticOutput, LayerList


class ActorCriticModel(th.nn.Module):
    """TODO."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList,
                 seed: int=123, dtype: str = "float32") -> None:
        """TODO."""
        super().__init__()
        self.base = BaseModel(lr, base_layer_list, base_activation_list, seed, dtype)
        self.policy_head = BaseModel(lr, policy_layer_list, policy_activation_list, seed, dtype)
        self.value_head = BaseModel(lr, value_layer_list, value_activation_list, seed, dtype)

    def forward(self, state: th.Tensor) -> ActorCriticOutput:
        """TODO."""
        x = self.base(state)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
