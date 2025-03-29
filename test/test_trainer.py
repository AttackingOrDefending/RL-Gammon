"""Tests for the verifying the trainer functionality."""
import numpy as np
import pytest

from rlgammon.buffers import UniformBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import Input, MovePart
from rlgammon.trainer.step_trainer import StepTrainer


@pytest.fixture
def buffer() -> UniformBuffer:
    """
    Initialize an empty buffer.

    :return: empty buffer
    """
    capacity = 10_000
    env = BackgammonEnv()
    return UniformBuffer(env.observation_shape, env.action_shape, capacity)


@pytest.fixture
def episode_buffer() -> list[tuple[Input, Input, MovePart, bool, int, int]]:
    """Initialize an episode buffer with pre-added data."""
    env = BackgammonEnv()
    episode_buffer: list[tuple[Input, Input, MovePart, bool, int, int]] = []

    state1 = env.get_input()
    action1 = (4, (20, 24))
    env.flip()
    next_state1 = env.get_input()
    done1 = False
    episode_buffer.append((state1, next_state1, action1[1], done1, 1, -1))

    env.flip()

    state2 = env.get_input()
    action2 = (6, (12, 18))
    env.flip()
    next_state2 = env.get_input()
    done2 = True
    episode_buffer.append((state2, next_state2, action2[1], done2, -1, 1))
    return episode_buffer


def test_load_parameters_valid() -> None:
    """Test that valid parameters are loaded."""
    trainer = StepTrainer()
    trainer.load_parameters("test_parameters/valid_test_parameters.json")
    assert trainer.is_ready_for_training()


def test_load_parameters_invalid() -> None:
    """Test that invalid parameters results in an error."""
    trainer = StepTrainer()
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        trainer.load_parameters("test_parameters/invalid_type_test_parameters.json")
    assert excinfo.type is ValueError


def test_finalize_data_win(episode_buffer: list[tuple[Input, Input, MovePart, bool, int, int]], buffer: UniformBuffer) -> None:
    """
    Test finalize data when someone wins.

    :param episode_buffer: Pre-initialized episode buffer containing episode observations.
    :param buffer: empty buffer where finalized samples will be recorded.
    """
    trainer = StepTrainer()

    # Create temporary parameters for the sake of testing
    trainer.parameters = {"decay": 0.99}

    losing_player = 1
    final_reward = 1

    trainer.finalize_data(episode_buffer, losing_player, final_reward, buffer)

    assert buffer.reward_buffer[0] == final_reward                                 # reward for player -1 (winning)
    assert buffer.reward_buffer[1] == -final_reward                                  # reward for player 1 (losing)
    assert np.array_equal(buffer.state_buffer[0], episode_buffer[0][1])
    assert np.array_equal(buffer.state_buffer[1], episode_buffer[0][0])
    assert np.array_equal(buffer.new_state_buffer[0], episode_buffer[1][1])
    assert np.array_equal(buffer.new_state_buffer[1], episode_buffer[1][0])


def test_finalize_data_draw(episode_buffer: list[tuple[Input, Input, MovePart, bool, int, int]],
                            buffer: UniformBuffer) -> None:
    """
    Test finalize data when there's a draw.

    :param episode_buffer: Pre-initialized episode buffer containing episode observations.
    :param buffer: empty buffer where finalized samples will be recorded.
    """
    trainer = StepTrainer()

    # Create temporary parameters for the sake of testing
    trainer.parameters = {"decay": 0.99}

    losing_player = 0
    final_reward = 0

    trainer.finalize_data(episode_buffer, losing_player, final_reward, buffer)
    assert buffer.reward_buffer[0] == 0
    assert buffer.reward_buffer[1] == 0
    assert np.array_equal(buffer.state_buffer[0], episode_buffer[0][1])
    assert np.array_equal(buffer.state_buffer[1], episode_buffer[0][0])
    assert np.array_equal(buffer.new_state_buffer[0], episode_buffer[1][1])
    assert np.array_equal(buffer.new_state_buffer[1], episode_buffer[1][0])
