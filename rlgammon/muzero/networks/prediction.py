"""Prediction network ``f`` of the Stochastic MuZero agent."""
import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp


class PredictionNetwork(nn.Module):
    """Predict the policy logits and the categorical value from a latent state."""

    def __init__(self, state_channels: int, hidden_sizes: tuple[int, ...], num_actions: int,
                 value_support_size: int) -> None:
        """
        Construct the policy and value heads sharing a common MLP trunk.

        :param state_channels: dimensionality of the input latent state
        :param hidden_sizes: widths of the hidden layers of the shared trunk
        :param num_actions: number of environment actions (policy head width)
        :param value_support_size: number of categorical atoms of the value head
        """
        super().__init__()
        self.policy_head = build_mlp(state_channels, hidden_sizes, num_actions)
        self.value_head = build_mlp(state_channels, hidden_sizes, value_support_size)

    def forward(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Predict the policy logits and value logits for a latent state.

        :param state: latent state tensor of shape ``[B, state_channels]``
        :return: a tuple ``(policy_logits [B, num_actions], value_logits [B, value_support_size])``
        """
        return self.policy_head(state), self.value_head(state)
