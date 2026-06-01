"""Full Stochastic MuZero network bundling the representation, prediction, dynamics and chance models."""
import torch as th
from torch import nn

from rlgammon.muzero.muzero_types import AfterstateOutput, MuZeroConfig, NetworkOutput
from rlgammon.muzero.networks.afterstate_dynamics import AfterstateDynamics
from rlgammon.muzero.networks.afterstate_prediction import AfterstatePrediction
from rlgammon.muzero.networks.chance_encoder import ChanceEncoder
from rlgammon.muzero.networks.dynamics import DynamicsNetwork
from rlgammon.muzero.networks.prediction import PredictionNetwork
from rlgammon.muzero.networks.representation import RepresentationNetwork
from rlgammon.muzero.networks.value_encoding import support_to_scalar


class StochasticMuZeroNetwork(nn.Module):
    """
    Bundle every sub-network of the Stochastic MuZero agent behind a single inference interface.

    The network exposes the three inference entry points used by the search:

    * :meth:`initial_inference` runs the representation ``h`` then the prediction ``f``.
    * :meth:`recurrent_inference_afterstate` runs the afterstate dynamics then the afterstate
      prediction, modelling the agent's deterministic action transition.
    * :meth:`recurrent_inference_state` runs the dynamics ``g`` then the prediction ``f``, modelling
      the stochastic chance transition from an afterstate to the next state.
    """

    def __init__(self, config: MuZeroConfig) -> None:
        """
        Construct every sub-network from a single :class:`MuZeroConfig`.

        :param config: configuration providing all network dimensions and hyper-parameters
        """
        super().__init__()
        self.config = config
        self.representation = RepresentationNetwork(
            config.observation_size, config.hidden_sizes, config.state_channels,
        )
        self.prediction = PredictionNetwork(
            config.state_channels, config.hidden_sizes, config.num_actions, config.value_support_size,
        )
        self.afterstate_dynamics = AfterstateDynamics(
            config.state_channels, config.num_actions, config.hidden_sizes,
        )
        self.afterstate_prediction = AfterstatePrediction(
            config.state_channels, config.hidden_sizes, config.codebook_size, config.value_support_size,
        )
        self.dynamics = DynamicsNetwork(
            config.state_channels, config.codebook_size, config.hidden_sizes, config.reward_support_size,
        )
        self.chance_encoder = ChanceEncoder(
            config.observation_size, config.hidden_sizes, config.codebook_size, config.commitment_cost,
            config.codebook_diversity_cost,
        )
        # Move every parameter onto the configured device so all sub-networks share one device; the
        # default ``"cpu"`` reproduces the original behaviour exactly.
        self.to(th.device(config.device))

    @property
    def device(self) -> th.device:
        """
        Return the device the network's parameters currently live on.

        :return: the torch device of the first parameter (the whole network shares one device)
        """
        return next(self.parameters()).device

    def initial_inference(self, observation: th.Tensor) -> NetworkOutput:
        """
        Run the representation then prediction networks on a raw observation.

        The reward at the root is defined as a zero scalar (there is no transition into the root),
        returned as categorical logits of the configured reward support size.

        :param observation: tensor of shape ``[B, observation_size]``
        :return: a :class:`NetworkOutput` with ``value`` / ``reward`` logits, ``policy_logits`` and ``state``
        """
        state = self.representation(observation)
        policy_logits, value_logits = self.prediction(state)
        reward_logits = th.zeros(
            (observation.shape[0], self.config.reward_support_size), dtype=state.dtype, device=state.device,
        )
        return NetworkOutput(value=value_logits, reward=reward_logits, policy_logits=policy_logits, state=state)

    def recurrent_inference_afterstate(self, state: th.Tensor, action_onehot: th.Tensor) -> AfterstateOutput:
        """
        Run the afterstate dynamics then afterstate prediction for a state-action pair.

        :param state: latent state tensor of shape ``[B, state_channels]``
        :param action_onehot: one-hot action tensor of shape ``[B, num_actions]``
        :return: an :class:`AfterstateOutput` with ``chance_logits``, ``afterstate_value`` logits and ``afterstate``
        """
        afterstate = self.afterstate_dynamics(state, action_onehot)
        chance_logits, afterstate_value_logits = self.afterstate_prediction(afterstate)
        return AfterstateOutput(
            chance_logits=chance_logits, afterstate_value=afterstate_value_logits, afterstate=afterstate,
        )

    def recurrent_inference_state(self, afterstate: th.Tensor, chance_onehot: th.Tensor) -> NetworkOutput:
        """
        Run the dynamics then prediction networks for an afterstate-chance pair.

        :param afterstate: afterstate tensor of shape ``[B, state_channels]``
        :param chance_onehot: one-hot chance outcome tensor of shape ``[B, codebook_size]``
        :return: a :class:`NetworkOutput` with ``value`` / ``reward`` logits, ``policy_logits`` and ``state``
        """
        next_state, reward_logits = self.dynamics(afterstate, chance_onehot)
        policy_logits, value_logits = self.prediction(next_state)
        return NetworkOutput(
            value=value_logits, reward=reward_logits, policy_logits=policy_logits, state=next_state,
        )

    def encode_chance(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode an observation into a straight-through one-hot chance code via the chance encoder.

        :param observation: tensor of shape ``[B, observation_size]``
        :return: a tuple ``(onehot_st [B, codebook_size], code_indices [B], commitment_loss scalar)``
        """
        # nn.Module.__call__ is typed as returning Any; the ChanceEncoder.forward signature is exact.
        return self.chance_encoder(observation)  # type: ignore[no-any-return]

    def value_to_scalar(self, value_logits: th.Tensor) -> th.Tensor:
        """
        Decode categorical value logits to a scalar value.

        :param value_logits: tensor of shape ``[B, value_support_size]`` of log-probabilities
        :return: scalar value tensor of shape ``[B]``
        """
        return support_to_scalar(value_logits, self.config.value_support_size)

    def reward_to_scalar(self, reward_logits: th.Tensor) -> th.Tensor:
        """
        Decode categorical reward logits to a scalar reward.

        :param reward_logits: tensor of shape ``[B, reward_support_size]`` of log-probabilities
        :return: scalar reward tensor of shape ``[B]``
        """
        return support_to_scalar(reward_logits, self.config.reward_support_size)
