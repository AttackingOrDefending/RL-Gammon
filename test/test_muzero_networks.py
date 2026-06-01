"""Tests for the Stochastic MuZero neural networks and value encoding."""
import numpy as np
import torch as th

from rlgammon.muzero.muzero_types import AfterstateOutput, MuZeroConfig, NetworkOutput
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.networks.value_encoding import scalar_to_support, support_to_scalar

# Small dimensions keeping the test networks tiny and fast.
BATCH_SIZE = 2
STATE_CHANNELS = 16
CODEBOOK_SIZE = 4
VALUE_SUPPORT_SIZE = 5
REWARD_SUPPORT_SIZE = 5
NUM_ACTIONS = 8
OBSERVATION_SIZE = 10


def _build_config() -> MuZeroConfig:
    """
    Build a small MuZeroConfig suitable for fast unit tests.

    :return: a MuZeroConfig with tiny network dimensions
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        state_channels=STATE_CHANNELS,
        hidden_sizes=(16,),
        codebook_size=CODEBOOK_SIZE,
        value_support_size=VALUE_SUPPORT_SIZE,
        reward_support_size=REWARD_SUPPORT_SIZE,
    )


def test_initial_inference_shapes() -> None:
    """Test that the initial inference produces a NetworkOutput with the expected tensor shapes."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    observation = th.zeros((BATCH_SIZE, OBSERVATION_SIZE))

    output = network.initial_inference(observation)

    assert isinstance(output, NetworkOutput)
    assert output.state.shape == (BATCH_SIZE, STATE_CHANNELS)
    assert output.policy_logits.shape == (BATCH_SIZE, NUM_ACTIONS)
    assert output.value.shape == (BATCH_SIZE, VALUE_SUPPORT_SIZE)
    assert output.reward.shape == (BATCH_SIZE, REWARD_SUPPORT_SIZE)


def test_recurrent_inference_afterstate_shapes() -> None:
    """Test that the afterstate inference produces an AfterstateOutput with the expected shapes."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    state = th.zeros((BATCH_SIZE, STATE_CHANNELS))
    action_onehot = th.zeros((BATCH_SIZE, NUM_ACTIONS))
    action_onehot[:, 0] = 1.0

    output = network.recurrent_inference_afterstate(state, action_onehot)

    assert isinstance(output, AfterstateOutput)
    assert output.afterstate.shape == (BATCH_SIZE, STATE_CHANNELS)
    assert output.chance_logits.shape == (BATCH_SIZE, CODEBOOK_SIZE)
    assert output.afterstate_value.shape == (BATCH_SIZE, VALUE_SUPPORT_SIZE)


def test_recurrent_inference_state_shapes() -> None:
    """Test that the state inference produces a NetworkOutput with the expected tensor shapes."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    afterstate = th.zeros((BATCH_SIZE, STATE_CHANNELS))
    chance_onehot = th.zeros((BATCH_SIZE, CODEBOOK_SIZE))
    chance_onehot[:, 0] = 1.0

    output = network.recurrent_inference_state(afterstate, chance_onehot)

    assert isinstance(output, NetworkOutput)
    assert output.state.shape == (BATCH_SIZE, STATE_CHANNELS)
    assert output.policy_logits.shape == (BATCH_SIZE, NUM_ACTIONS)
    assert output.value.shape == (BATCH_SIZE, VALUE_SUPPORT_SIZE)
    assert output.reward.shape == (BATCH_SIZE, REWARD_SUPPORT_SIZE)


def test_chance_encoder_one_hot() -> None:
    """Test that the chance encoder yields valid one-hot codes, in-range indices and a non-negative loss."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    observation = th.randn((BATCH_SIZE, OBSERVATION_SIZE))

    onehot_st, code_indices, commitment_loss = network.encode_chance(observation)

    assert onehot_st.shape == (BATCH_SIZE, CODEBOOK_SIZE)
    # Each row is a valid one-hot: it sums to one and has exactly one active entry.
    assert th.allclose(onehot_st.sum(dim=1), th.ones(BATCH_SIZE))
    assert th.all((onehot_st == 1.0).sum(dim=1) == 1)
    # The hard one-hot must agree with the reported code indices.
    assert th.all(th.argmax(onehot_st, dim=1) == code_indices)
    assert th.all(code_indices >= 0)
    assert th.all(code_indices < CODEBOOK_SIZE)
    # The commitment loss is a non-negative scalar.
    assert commitment_loss.ndim == 0
    assert commitment_loss.item() >= 0.0


def test_chance_encoder_straight_through_gradients() -> None:
    """Test that gradients reach the chance encoder parameters through the straight-through code."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    observation = th.randn((BATCH_SIZE, OBSERVATION_SIZE))

    onehot_st, _, _ = network.encode_chance(observation)
    # A plain ``.sum()`` cancels (softmax rows sum to 1), so weight the classes
    # non-uniformly to get a genuine straight-through gradient back to the encoder.
    class_weights = th.arange(1, onehot_st.shape[-1] + 1, dtype=onehot_st.dtype)
    (onehot_st * class_weights).sum().backward()  # type: ignore[no-untyped-call]

    first_parameter = next(iter(network.chance_encoder.encoder.parameters()))
    assert first_parameter.grad is not None
    assert th.any(first_parameter.grad != 0.0)


def _distinct_codes_after_training(diversity_cost: float) -> int:
    """
    Run the real Stochastic-MuZero training loop on diverse data and count the codes the encoder uses.

    Several short trajectories with DISTINCT observations are stored and the full
    :func:`~rlgammon.muzero.training.losses.muzero_loss` is optimized through the learner, reproducing
    the exact chance-loss dynamics of a real run. With ``diversity_cost == 0`` (the original encoder)
    the degenerate optimum collapses the codebook onto a single code; a non-zero ``diversity_cost``
    must keep many codes in use so the learned dynamics can depend on the chance outcome.

    :param diversity_cost: weight of the codebook-diversity regularizer to exercise
    :return: the number of distinct codes the encoder assigns to a fixed diverse batch after training
    """
    from rlgammon.muzero.replay.replay_buffer import MuZeroReplayBuffer  # noqa: PLC0415
    from rlgammon.muzero.replay.trajectory import Step, Trajectory  # noqa: PLC0415
    from rlgammon.muzero.training.learner import MuZeroLearner  # noqa: PLC0415

    codebook_size = 8
    observation_size = 20
    config = MuZeroConfig(
        observation_size=observation_size, num_actions=NUM_ACTIONS, state_channels=STATE_CHANNELS,
        hidden_sizes=(32,), codebook_size=codebook_size, unroll_steps=2, td_steps=5, batch_size=16,
        value_support_size=VALUE_SUPPORT_SIZE, reward_support_size=REWARD_SUPPORT_SIZE,
        codebook_diversity_cost=diversity_cost, seed=0,
    )
    th.manual_seed(0)
    network = StochasticMuZeroNetwork(config)
    buffer = MuZeroReplayBuffer(config)
    rng = np.random.default_rng(0)
    for game in range(8):
        steps = [
            Step(
                observation=list(np.random.default_rng(game * 10 + index).normal(size=observation_size)),
                action=index % NUM_ACTIONS, reward=0.0, policy={index % NUM_ACTIONS: 1.0},
                value=0.0, to_play=index % 2,
            )
            for index in range(4)
        ]
        steps[-1].reward = 1.0
        buffer.save(Trajectory(steps=steps, returns=[1.0, -1.0]))

    learner = MuZeroLearner(config, network)
    for _ in range(400):
        learner.train_step(buffer.sample_batch(rng))

    network.eval()
    observations = th.tensor(
        [list(np.random.default_rng(1000 + index).normal(size=observation_size)) for index in range(128)],
        dtype=th.float32,
    )
    with th.no_grad():
        _, code_indices, _ = network.encode_chance(observations)
    return len({int(code) for code in code_indices})


def test_codebook_diversity_prevents_collapse() -> None:
    """Test that the codebook-diversity term keeps many codes in use where the bare encoder collapses."""
    codebook_size = 8
    collapsed = _distinct_codes_after_training(diversity_cost=0.0)
    diverse = _distinct_codes_after_training(diversity_cost=1.0)

    # The bare encoder under-uses the codebook (on real high-dimensional observations it collapses
    # all the way to a single code, vanishing the chance loss and commitment); the diversity term must
    # keep (nearly) the whole codebook in use and strictly more codes than the bare encoder, so the
    # learned dynamics can depend on the sampled chance outcome.
    assert collapsed < codebook_size, f"bare encoder should under-use the codebook, used {collapsed}/{codebook_size}"
    assert diverse >= codebook_size - 1, (
        f"diversity term should use (nearly) the whole codebook, used {diverse}/{codebook_size}"
    )
    assert diverse > collapsed, f"diversity must increase code usage: {collapsed} -> {diverse}"


def test_value_encoding_round_trip() -> None:
    """Test that decoding the log of the two-hot encoding recovers the original scalar."""
    scalars = th.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])

    support = scalar_to_support(scalars, VALUE_SUPPORT_SIZE)
    recovered = support_to_scalar(th.log(support + 1e-12), VALUE_SUPPORT_SIZE)

    assert th.allclose(recovered, scalars, atol=1e-3)
