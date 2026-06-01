"""File storing types associated with the Stochastic MuZero agent."""
from dataclasses import dataclass
from enum import Enum

import torch as th


@dataclass(frozen=True)
class MuZeroConfig:
    """Immutable configuration bundling every hyper-parameter of the Stochastic MuZero agent."""

    observation_size: int = 198
    num_actions: int = 1352
    state_channels: int = 256
    hidden_sizes: tuple[int, ...] = (256, 256)
    codebook_size: int = 32
    num_simulations: int = 100
    # Simulation budget used at EVALUATION time (kept separate from the self-play training budget).
    eval_num_simulations: int = 200
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.25
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 1.0
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    chance_loss_weight: float = 1.0
    commitment_cost: float = 0.25
    # Weight of the codebook-diversity (load-balancing) regularizer that prevents the VQ chance
    # encoder from collapsing to a single code. It maximizes the entropy of the batch-averaged soft
    # code assignment; ``0.0`` reproduces the original (collapse-prone) behaviour. See
    # :class:`~rlgammon.muzero.networks.chance_encoder.ChanceEncoder`.
    codebook_diversity_cost: float = 1.0
    value_support_size: int = 21
    reward_support_size: int = 21
    replay_capacity: int = 100_000
    seed: int = 123
    # Torch device the network and all inference/training tensors live on: ``"cpu"`` (default,
    # reproducing the original behaviour) or ``"cuda"`` to use an NVIDIA GPU. Selection of ``"cuda"``
    # is guarded by :func:`torch.cuda.is_available` at the call sites, which fall back to ``"cpu"``.
    device: str = "cpu"


@dataclass
class NetworkOutput:
    """Bundle of the tensors produced by an inference step that yields a full latent state."""

    value: th.Tensor
    reward: th.Tensor
    policy_logits: th.Tensor
    state: th.Tensor


@dataclass
class AfterstateOutput:
    """Bundle of the tensors produced by an inference step that yields an afterstate."""

    chance_logits: th.Tensor
    afterstate_value: th.Tensor
    afterstate: th.Tensor


class PossibleMuZero(Enum):
    """Enumeration of the supported MuZero variants."""

    STOCHASTIC = "SMZ"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleMuZero":
        """
        Convert a string, found e.g. in JSON parameters, to a PossibleMuZero enum.

        :param string_to_convert: the string value to convert
        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "SMZ":
                return PossibleMuZero.STOCHASTIC
            case _:
                return None  # type: ignore[return-value]
