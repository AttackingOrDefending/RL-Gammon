"""Dynamics network ``g`` of the Stochastic MuZero agent."""
import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp, normalize_latent


class DynamicsNetwork(nn.Module):
    """Map an afterstate and a chance outcome to the next latent state and the reward."""

    def __init__(self, state_channels: int, codebook_size: int, hidden_sizes: tuple[int, ...],
                 reward_support_size: int) -> None:
        """
        Construct the next-state and reward heads of the dynamics network.

        :param state_channels: dimensionality of the afterstate and produced next state
        :param codebook_size: number of chance outcomes (width of the chance one-hot)
        :param hidden_sizes: widths of the hidden layers of the MLP bodies
        :param reward_support_size: number of categorical atoms of the reward head
        """
        super().__init__()
        self.state_head = build_mlp(state_channels + codebook_size, hidden_sizes, state_channels)
        self.reward_head = build_mlp(state_channels + codebook_size, hidden_sizes, reward_support_size)

    def forward(self, afterstate: th.Tensor, chance_onehot: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Map an afterstate and a one-hot chance outcome to the next state and reward logits.

        :param afterstate: afterstate tensor of shape ``[B, state_channels]``
        :param chance_onehot: one-hot chance outcome tensor of shape ``[B, codebook_size]``
        :return: a tuple ``(next_state [B, state_channels], reward_logits [B, reward_support_size])``
        """
        joined = th.cat([afterstate, chance_onehot], dim=1)
        next_state = normalize_latent(self.state_head(joined))
        return next_state, self.reward_head(joined)
