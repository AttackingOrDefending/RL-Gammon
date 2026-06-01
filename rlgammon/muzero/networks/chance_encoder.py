"""Chance encoder ``e`` of the Stochastic MuZero agent (VQ-VAE style)."""
import math

import torch as th
from torch import nn

from rlgammon.muzero.networks.mlp import build_mlp

# Floor added inside logarithms when computing the batch-mean assignment entropy, avoiding ``log(0)``
# for codes that receive no soft mass in a batch.
_ENTROPY_EPS = 1e-9


class ChanceEncoder(nn.Module):
    """
    Encode an observation into a one-hot chance code with a straight-through estimator.

    Following the Stochastic MuZero paper, the encoder produces ``codebook_size`` logits which are
    discretized to a hard one-hot vector via ``argmax``. Rather than maintaining an explicit
    embedding codebook, the ``codebook_size`` logits are treated directly as the code: the forward
    code is the hard one-hot, while gradients are routed to the encoder through the softmax
    probabilities using the straight-through trick ``onehot_st = onehot + (probs - probs.detach())``.

    Two regularizers are returned (bundled into a single scalar):

    * a **commitment** loss ``commitment_cost * MSE(probs, onehot.detach())`` keeping the encoder's
      soft distribution close to its own discretization;
    * a **codebook-diversity** (load-balancing) loss ``diversity_cost * (log(K) - H(mean_b probs))``
      that penalizes a low-entropy batch-averaged assignment. Without it the trivial optimum of the
      chance objective is to map EVERY observation to one code (the chance head then predicts that
      single code with zero cross-entropy and commitment collapses to zero too): the codebook
      collapses to a single entry and the learned dynamics ignores the dice stochasticity entirely.
      Maximizing the entropy of the mean soft assignment ``p_bar = mean_b softmax(logits_b)`` is the
      standard VQ / mixture load-balancing fix and spreads usage across the codes. The term is offset
      by ``log(codebook_size)`` so it is non-negative and reads ``0`` at perfectly uniform usage.
    """

    def __init__(self, observation_size: int, hidden_sizes: tuple[int, ...], codebook_size: int,
                 commitment_cost: float, diversity_cost: float = 0.0) -> None:
        """
        Construct the chance encoder MLP.

        :param observation_size: dimensionality of the raw observation vector
        :param hidden_sizes: widths of the hidden layers of the MLP body
        :param codebook_size: number of chance outcomes (width of the produced code)
        :param commitment_cost: weight of the commitment loss term
        :param diversity_cost: weight of the codebook-diversity (entropy load-balancing) term; the
            default ``0.0`` reproduces the original commitment-only (collapse-prone) behaviour
        """
        super().__init__()
        self.encoder = build_mlp(observation_size, hidden_sizes, codebook_size)
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.diversity_cost = diversity_cost
        # Maximum achievable entropy of the code distribution (uniform usage), used to turn the
        # entropy-maximization into a non-negative, minimized loss.
        self._max_entropy = math.log(codebook_size)

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode an observation into a straight-through one-hot chance code.

        :param observation: tensor of shape ``[B, observation_size]``
        :return: a tuple ``(onehot_st [B, codebook_size], code_indices [B], vq_loss scalar)`` where
            ``onehot_st`` rows are one-hot (carrying gradients to the encoder), ``code_indices`` holds
            the selected atom per row in ``[0, codebook_size)`` and ``vq_loss`` is the non-negative
            sum of the commitment and codebook-diversity regularizers (already weighted by their
            respective costs)
        """
        logits = self.encoder(observation)
        probabilities = th.softmax(logits, dim=1)

        code_indices = th.argmax(probabilities, dim=1)
        onehot = th.nn.functional.one_hot(code_indices, num_classes=self.codebook_size).to(probabilities.dtype)

        # Straight-through estimator: forward value is the hard one-hot, gradient flows via probs.
        onehot_st = onehot + (probabilities - probabilities.detach())
        commitment_loss = self.commitment_cost * th.mean((probabilities - onehot.detach()) ** 2)

        # Codebook-diversity (load-balancing): maximize the entropy of the batch-mean soft assignment
        # so the encoder is pushed to use many codes instead of collapsing onto one.
        mean_assignment = probabilities.mean(dim=0)
        mean_entropy = -th.sum(mean_assignment * th.log(mean_assignment + _ENTROPY_EPS))
        diversity_loss = self.diversity_cost * (self._max_entropy - mean_entropy)

        return onehot_st, code_indices, commitment_loss + diversity_loss
