"""Tests for the Stochastic MuZero K-step unrolled loss and the learner."""
import math

import numpy as np
import torch as th

from rlgammon.game.mock_game import MockGame
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.networks.value_encoding import scalar_to_support, support_to_scalar
from rlgammon.muzero.replay.replay_buffer import Batch, MuZeroReplayBuffer
from rlgammon.muzero.self_play.actor import SelfPlayActor
from rlgammon.muzero.training.learner import MuZeroLearner
from rlgammon.muzero.training.losses import muzero_loss

# Tiny dimensions keeping the loss / learner tests fast and pyspiel-free.
UNROLL_STEPS = 3
NUM_ACTIONS = 8
OBSERVATION_SIZE = 198
STATE_CHANNELS = 16
HIDDEN_SIZES = (16,)
CODEBOOK_SIZE = 2
VALUE_SUPPORT_SIZE = 3
REWARD_SUPPORT_SIZE = 3
BATCH_SIZE = 4
NUM_SIMULATIONS = 4
# A fixed seed so the self-play that fills the buffer is deterministic.
SEED = 11
# The exact set of loss keys the loss dict must expose.
LOSS_KEYS = {"total", "value", "policy", "reward", "chance", "commitment"}
# Number of self-play games used to fill the buffer.
FILL_GAMES = 3
# Number of learner train steps exercised by the smoke test.
SMOKE_TRAIN_STEPS = 3
# Tolerance for the weighted-sum and round-trip allclose checks.
TOLERANCE = 1e-4
ROUND_TRIP_TOLERANCE = 1e-3


def _build_config() -> MuZeroConfig:
    """
    Build a tiny MuZeroConfig sized for the loss and learner tests.

    :return: a small :class:`MuZeroConfig`
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        state_channels=STATE_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        codebook_size=CODEBOOK_SIZE,
        num_simulations=NUM_SIMULATIONS,
        unroll_steps=UNROLL_STEPS,
        td_steps=5,
        batch_size=BATCH_SIZE,
        value_support_size=VALUE_SUPPORT_SIZE,
        reward_support_size=REWARD_SUPPORT_SIZE,
    )


def _fill_buffer(config: MuZeroConfig, network: StochasticMuZeroNetwork) -> MuZeroReplayBuffer:
    """
    Play a few self-play games on the mock game and store them in a fresh replay buffer.

    :param config: the configuration shared with self-play and the buffer
    :param network: the network driving self-play
    :return: a replay buffer holding the played trajectories
    """
    rng = np.random.default_rng(SEED)
    actor = SelfPlayActor(config, MockGame(), network, rng)
    buffer = MuZeroReplayBuffer(config)
    for _ in range(FILL_GAMES):
        buffer.save(actor.play_game())
    return buffer


def _sample_batch(config: MuZeroConfig, network: StochasticMuZeroNetwork) -> Batch:
    """
    Fill a buffer from self-play and sample a single training batch from it.

    :param config: the configuration shared with self-play, the buffer and sampling
    :param network: the network driving self-play
    :return: a sampled :class:`Batch`
    """
    buffer = _fill_buffer(config, network)
    return buffer.sample_batch(np.random.default_rng(SEED))


def test_loss_has_all_keys_and_finite_scalars() -> None:
    """Test that the loss dict exposes exactly the six keys, each a finite scalar tensor."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    batch = _sample_batch(config, network)

    losses = muzero_loss(config, network, batch)

    assert set(losses) == LOSS_KEYS
    for value in losses.values():
        assert value.ndim == 0
        assert math.isfinite(float(value))


def test_total_is_the_weighted_sum() -> None:
    """Test that ``total`` is the configured weighted sum of the component losses."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    batch = _sample_batch(config, network)

    losses = muzero_loss(config, network, batch)

    expected = (
        config.value_loss_weight * losses["value"]
        + config.policy_loss_weight * losses["policy"]
        + config.reward_loss_weight * losses["reward"]
        + config.chance_loss_weight * losses["chance"]
        + losses["commitment"]
    )
    assert th.allclose(losses["total"], expected, atol=TOLERANCE)


def test_backward_reaches_every_subnetwork() -> None:
    """Test that ``total.backward()`` produces gradients on every sub-network's parameters."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    batch = _sample_batch(config, network)

    losses = muzero_loss(config, network, batch)
    losses["total"].backward()  # type: ignore[no-untyped-call]

    submodules = {
        "representation": network.representation,
        "prediction": network.prediction,
        "dynamics": network.dynamics,
        "afterstate_dynamics": network.afterstate_dynamics,
        "afterstate_prediction": network.afterstate_prediction,
        "chance_encoder": network.chance_encoder,
    }
    for name, module in submodules.items():
        grads = [parameter.grad for parameter in module.parameters()]
        assert grads, f"{name} has no parameters"
        assert all(grad is not None for grad in grads), f"{name} has a None gradient"
        assert any(th.any(grad != 0.0) for grad in grads if grad is not None), f"{name} has only zero gradients"


def test_value_encoding_round_trip() -> None:
    """Test that decoding the log of the two-hot encoding recovers the original scalar."""
    scalars = th.tensor([-1.0, -0.25, 0.0, 0.5, 1.0])

    support = scalar_to_support(scalars, VALUE_SUPPORT_SIZE)
    recovered = support_to_scalar(th.log(support + 1e-12), VALUE_SUPPORT_SIZE)

    assert th.allclose(recovered, scalars, atol=ROUND_TRIP_TOLERANCE)


def test_learner_train_steps_return_finite_losses() -> None:
    """Test that a few learner train steps run and return finite component losses."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    buffer = _fill_buffer(config, network)
    learner = MuZeroLearner(config, network)
    rng = np.random.default_rng(SEED)

    for _ in range(SMOKE_TRAIN_STEPS):
        batch = buffer.sample_batch(rng)
        losses = learner.train_step(batch)
        assert set(losses) == LOSS_KEYS
        assert all(math.isfinite(value) for value in losses.values())
