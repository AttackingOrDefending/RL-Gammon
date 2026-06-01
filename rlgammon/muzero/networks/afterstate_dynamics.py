"""Afterstate dynamics network of the Stochastic MuZero agent."""
import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp, normalize_latent


class AfterstateDynamics(nn.Module):
    """Map a latent state and an action to the deterministic afterstate ``as``."""

    def __init__(self, state_channels: int, num_actions: int, hidden_sizes: tuple[int, ...]) -> None:
        """
        Construct the afterstate dynamics MLP.

        :param state_channels: dimensionality of the latent state and produced afterstate
        :param num_actions: number of environment actions (width of the action one-hot)
        :param hidden_sizes: widths of the hidden layers of the MLP body
        """
        super().__init__()
        self.body = build_mlp(state_channels + num_actions, hidden_sizes, state_channels)

    def forward(self, state: th.Tensor, action_onehot: th.Tensor) -> th.Tensor:
        """
        Map a latent state and a one-hot action to the normalized afterstate.

        :param state: latent state tensor of shape ``[B, state_channels]``
        :param action_onehot: one-hot action tensor of shape ``[B, num_actions]``
        :return: normalized afterstate of shape ``[B, state_channels]``
        """
        joined = th.cat([state, action_onehot], dim=1)
        return normalize_latent(self.body(joined))
