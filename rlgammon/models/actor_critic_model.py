"""TODO."""
import torch as th

from rlgammon.models.model_types import ActivationList, ActorCriticOutput, LayerList
from rlgammon.models.raw_model import RawModel
from rlgammon.rlgammon_types import Feature


class ActorCriticModel(th.nn.Module):
    """TODO."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """TODO."""
        super().__init__()
        self.base = RawModel(base_layer_list, base_activation_list)
        self.policy_head = RawModel(policy_layer_list, policy_activation_list)
        self.value_head = RawModel(value_layer_list, value_activation_list)

        self.lr = lr
        self.lr_step_count = 100
        self.lr_step_current_counter = 0
        self.decay_rate = 0.96

        self.optimizer = th.optim.Adam(params=list(self.parameters()), lr=self.lr)
        self.lr_scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.decay_rate)

    def forward(self, state: Feature | th.Tensor) -> ActorCriticOutput:
        """TODO."""
        x = self.base(state)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
