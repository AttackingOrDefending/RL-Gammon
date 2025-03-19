"""Tests for the verifying the trainer functionality."""

import pytest

from rlgammon.buffers import UniformBuffer
from rlgammon.trainer.step_trainer import StepTrainer


@pytest.fixture
def init_buffer() -> UniformBuffer:
    buffer = UniformBuffer((2, 2), 2, 10)
    return buffer

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

@pytest.mark.parametrize("init_buffer", [])
def test_finalize_data_win(buffer: UniformBuffer) -> None:
    """
    Test that finalize data wins.

    :param buffer: TODO
    """
    trainer = StepTrainer()

    episode_buffer = []
    losing_player = 1
    final_reward = 1

    # TODO: Add data to test
    trainer.finalize_data(episode_buffer, losing_player, final_reward, buffer)
    assert 1 == 1

@pytest.mark.parametrize("init_buffer", [])
def test_finalize_data_draw(buffer: UniformBuffer) -> None:
    """
    Test that finalize data draws.

    :param buffer: TODO
    """
    trainer = StepTrainer()

    episode_buffer = []
    losing_player = -1
    final_reward = 0
    trainer.finalize_data(episode_buffer, losing_player, final_reward, buffer)

    assert 1 == 1
