"""Afterstate prediction network of the Stochastic MuZero agent."""
import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp


class AfterstatePrediction(nn.Module):
    """Predict the chance distribution ``sigma`` and the afterstate value ``Q`` from an afterstate."""

    def __init__(self, state_channels: int, hidden_sizes: tuple[int, ...], codebook_size: int,
                 value_support_size: int) -> None:
        """
        Construct the chance and afterstate-value heads sharing a common MLP trunk.

        :param state_channels: dimensionality of the input afterstate
        :param hidden_sizes: widths of the hidden layers of the shared trunk
        :param codebook_size: number of chance outcomes (chance head width)
        :param value_support_size: number of categorical atoms of the afterstate-value head
        """
        super().__init__()
        self.chance_head = build_mlp(state_channels, hidden_sizes, codebook_size)
        self.value_head = build_mlp(state_channels, hidden_sizes, value_support_size)

    def forward(self, afterstate: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Predict the chance logits and afterstate value logits for an afterstate.

        :param afterstate: afterstate tensor of shape ``[B, state_channels]``
        :return: a tuple ``(chance_logits [B, codebook_size], afterstate_value_logits [B, value_support_size])``
        """
        return self.chance_head(afterstate), self.value_head(afterstate)
