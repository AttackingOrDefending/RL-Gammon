"""Representation network ``h`` of the Stochastic MuZero agent."""
import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp, normalize_latent


class RepresentationNetwork(nn.Module):
    """Encode a raw observation into the initial latent state ``s_0``."""

    def __init__(self, observation_size: int, hidden_sizes: tuple[int, ...], state_channels: int) -> None:
        """
        Construct the representation MLP.

        :param observation_size: dimensionality of the raw observation vector
        :param hidden_sizes: widths of the hidden layers of the MLP body
        :param state_channels: dimensionality of the produced latent state
        """
        super().__init__()
        self.body = build_mlp(observation_size, hidden_sizes, state_channels)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """
        Map an observation to a normalized latent state.

        The latent is min-max normalized to ``[0, 1]`` over the feature dimension, as recommended
        in the MuZero appendix, to keep latent magnitudes bounded.

        :param observation: tensor of shape ``[B, observation_size]``
        :return: normalized latent state of shape ``[B, state_channels]``
        """
        return normalize_latent(self.body(observation))
