"""File implementing a generic actor-critic style model."""
import torch as th

from rlgammon.models.model_types import ActivationList, ActorCriticOutput, LayerList
from rlgammon.models.raw_model import RawModel
from rlgammon.rlgammon_types import Feature


class ActorCriticModel(th.nn.Module):
    """Class implementing a generic actor-critic style model."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """
        Construct a generic actor-critic style model by combining 3 networks: base, policy and value networks,
        and then initializing td specific parameters.

        :param lr: learning rate
        :param base_layer_list: list of layers to use in the base (shared) network
        :param base_activation_list: list of activations to use in the base (shared) network
        :param policy_layer_list: list of layers to use in the policy network
        :param policy_activation_list: list of activations to use in the policy network
        :param value_layer_list: list of layers to use in the value network
        :param value_activation_list: list of activations to use in the value network
        """
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

    def forward(self, x: Feature | th.Tensor) -> ActorCriticOutput:
        """
        Forward pass of the actor-critic style model.

        :param x: input data to the model
        :return: two separate outputs: value and policy for the input
        """
        # Process state - mcts gives one element list of raw observations (len = 200) from environment
        # Need the observation of len = 198 -> no dice
        base = self.base(x)
        p = self.policy_head(base)
        v = self.value_head(base) * 3
        return v, p
