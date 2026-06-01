"""Tests for the Stochastic MuZero replay buffer and its trajectory data contracts."""
import numpy as np
import torch as th

from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.replay.replay_buffer import Batch, MuZeroReplayBuffer
from rlgammon.muzero.replay.trajectory import Step, Trajectory

# Tiny replay configuration sized for the assertions below.
UNROLL_STEPS = 3
TD_STEPS = 4
BATCH_SIZE = 5
NUM_ACTIONS = 8
OBSERVATION_SIZE = 198
# A small capacity to exercise the oldest-trajectory eviction path.
SMALL_CAPACITY = 10
# Trajectory lengths used by the length / capacity tests.
FIRST_LENGTH = 4
SECOND_LENGTH = 3
COMBINED_LENGTH = FIRST_LENGTH + SECOND_LENGTH
CAPPED_LENGTH = 6
# Tolerance for the policy-rows-sum-to-one check.
SUM_TOLERANCE = 1e-5


def _build_config(replay_capacity: int = 1000) -> MuZeroConfig:
    """
    Build a tiny replay-sized configuration.

    :param replay_capacity: the step capacity of the buffer under test
    :return: a MuZeroConfig with small unroll/td/batch dimensions
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        unroll_steps=UNROLL_STEPS,
        td_steps=TD_STEPS,
        batch_size=BATCH_SIZE,
        replay_capacity=replay_capacity,
    )


def _make_step(action: int, reward: float, value: float, to_play: int) -> Step:
    """
    Build a single step with a one-hot-ish sparse policy on its action.

    :param action: the action played at this step (also the sole key of the policy)
    :param reward: the reward recorded after this step
    :param value: the MCTS root value recorded at this step
    :param to_play: the player to move at this step
    :return: a fully populated :class:`Step`
    """
    return Step(
        observation=[float(action)] * OBSERVATION_SIZE,
        action=action,
        reward=reward,
        policy={action: 1.0},
        value=value,
        to_play=to_play,
    )


def _make_trajectory(length: int) -> Trajectory:
    """
    Build a deterministic trajectory of a given length with a terminal reward on the last step.

    :param length: the number of decision steps in the trajectory
    :return: a populated :class:`Trajectory` whose last reward is 1.0
    """
    steps = [
        _make_step(action=index % 3 + 1, reward=0.0, value=0.1 * index, to_play=index % 2)
        for index in range(length)
    ]
    steps[-1].reward = 1.0
    return Trajectory(steps=steps, returns=[1.0, -1.0])


def test_batch_shapes_and_dtypes() -> None:
    """Test that a sampled batch has exactly the frozen field shapes and dtypes."""
    config = _build_config()
    buffer = MuZeroReplayBuffer(config)
    buffer.save(_make_trajectory(7))
    buffer.save(_make_trajectory(5))

    batch = buffer.sample_batch(np.random.default_rng(0))

    assert isinstance(batch, Batch)
    assert batch.observation.shape == (BATCH_SIZE, OBSERVATION_SIZE)
    assert batch.actions.shape == (BATCH_SIZE, UNROLL_STEPS)
    assert batch.target_values.shape == (BATCH_SIZE, UNROLL_STEPS + 1)
    assert batch.target_rewards.shape == (BATCH_SIZE, UNROLL_STEPS + 1)
    assert batch.target_policies.shape == (BATCH_SIZE, UNROLL_STEPS + 1, NUM_ACTIONS)
    assert batch.chance_observations.shape == (BATCH_SIZE, UNROLL_STEPS, OBSERVATION_SIZE)
    assert batch.weights.shape == (BATCH_SIZE,)

    assert batch.actions.dtype == th.long
    assert batch.observation.dtype == th.float32
    assert batch.target_values.dtype == th.float32
    assert batch.weights.dtype == th.float32


def test_target_policies_are_normalized() -> None:
    """Test that every dense policy row of a sampled batch sums to one over the action axis."""
    config = _build_config()
    buffer = MuZeroReplayBuffer(config)
    buffer.save(_make_trajectory(8))

    batch = buffer.sample_batch(np.random.default_rng(1))

    sums = batch.target_policies.sum(dim=-1)
    assert th.allclose(sums, th.ones_like(sums), atol=SUM_TOLERANCE)


def test_weights_are_ones() -> None:
    """Test that the importance weights are all ones for uniform sampling."""
    config = _build_config()
    buffer = MuZeroReplayBuffer(config)
    buffer.save(_make_trajectory(6))

    batch = buffer.sample_batch(np.random.default_rng(2))

    assert th.all(batch.weights == 1.0)


def test_len_reflects_stored_steps() -> None:
    """Test that the buffer length is the summed step count of the stored trajectories."""
    config = _build_config()
    buffer = MuZeroReplayBuffer(config)
    assert len(buffer) == 0

    buffer.save(_make_trajectory(FIRST_LENGTH))
    assert len(buffer) == FIRST_LENGTH
    buffer.save(_make_trajectory(SECOND_LENGTH))
    assert len(buffer) == COMBINED_LENGTH


def test_capacity_drops_oldest_trajectories() -> None:
    """Test that exceeding the step capacity evicts the oldest trajectories first."""
    config = _build_config(replay_capacity=SMALL_CAPACITY)
    buffer = MuZeroReplayBuffer(config)

    buffer.save(_make_trajectory(CAPPED_LENGTH))
    buffer.save(_make_trajectory(CAPPED_LENGTH))
    # The first trajectory must be evicted to keep the total at or below the capacity.
    assert len(buffer) == CAPPED_LENGTH
    buffer.save(_make_trajectory(CAPPED_LENGTH))
    assert len(buffer) == CAPPED_LENGTH
    assert len(buffer) <= SMALL_CAPACITY


def test_empty_trajectory_is_ignored() -> None:
    """Test that saving an empty trajectory leaves the buffer untouched."""
    config = _build_config()
    buffer = MuZeroReplayBuffer(config)

    buffer.save(Trajectory())

    assert len(buffer) == 0


def test_n_step_value_target() -> None:
    """
    Test the n-step bootstrapped value target on a hand-made trajectory.

    With ``discount == 1`` every value target is the discounted reward-to-go plus the stored value at
    ``index + td_steps`` when that step still lies inside the game, else the pure reward-to-go.
    """
    config = _build_config()
    # Ten steps, all rewards zero except a terminal reward of 1.0 on the last step.
    steps = [_make_step(action=1, reward=0.0, value=0.5, to_play=index % 2) for index in range(10)]
    steps[-1].reward = 1.0
    trajectory = Trajectory(steps=steps, returns=[1.0, -1.0])

    target = trajectory.make_target(start_index=0, config=config)

    # Step 0 bootstraps on the stored value at index td_steps (still inside the game): rewards-to-go are 0.
    assert target.target_values[0] == steps[TD_STEPS].value
    # The reward target for step 0 is the recorded reward at step 0 (zero here).
    assert target.target_rewards[0] == 0.0
    # The action targets read straight from the steps.
    assert target.actions == [1, 1, 1]


def test_absorbing_steps_past_terminal() -> None:
    """Test that windows running past the game's end yield zero value/reward and uniform policy."""
    config = _build_config()
    trajectory = _make_trajectory(2)

    # Start at the last real step so the window of unroll_steps + 1 runs off the end.
    target = trajectory.make_target(start_index=1, config=config)

    uniform = 1.0 / NUM_ACTIONS
    # Step index 2 and beyond are absorbing: zero value/reward and a uniform policy row.
    assert target.target_values[1] == 0.0
    assert target.target_rewards[1] == 0.0
    assert all(abs(probability - uniform) < SUM_TOLERANCE for probability in target.target_policies[1])
    # The following observations past the end repeat the last recorded observation.
    assert target.following_observations[-1] == trajectory.steps[-1].observation
