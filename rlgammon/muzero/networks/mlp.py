"""Small shared helpers for building the MLP bodies used across the MuZero networks."""
import torch as th
from torch import nn

# Smallest denominator allowed when min-max normalizing a latent, avoiding division by zero.
MIN_MAX_EPS = 1e-5


def build_mlp(input_size: int, hidden_sizes: tuple[int, ...], output_size: int) -> nn.Sequential:
    """
    Build a multi-layer perceptron with ``LayerNorm`` and ``ReLU`` between hidden layers.

    Each hidden layer is ``Linear -> LayerNorm -> ReLU``; the final layer is a bare ``Linear`` so
    callers can attach their own activation / interpret the outputs as logits.

    :param input_size: number of input features
    :param hidden_sizes: widths of the hidden layers (may be empty for a single linear map)
    :param output_size: number of output features
    :return: the assembled sequential MLP
    """
    layers: list[nn.Module] = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(previous_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, output_size))
    return nn.Sequential(*layers)


def normalize_latent(latent: th.Tensor) -> th.Tensor:
    """
    Scale a latent tensor to ``[0, 1]`` with a per-row min-max normalization (as in MuZero).

    The minimum and maximum are taken over the feature dimension so each example in the batch is
    rescaled independently, which keeps the latent magnitudes bounded across recurrent unrolls.

    :param latent: latent tensor of shape ``[B, channels]``
    :return: the min-max normalized latent of the same shape
    """
    minimum = latent.min(dim=1, keepdim=True).values
    maximum = latent.max(dim=1, keepdim=True).values
    scale = (maximum - minimum).clamp(min=MIN_MAX_EPS)
    return (latent - minimum) / scale
