"""Tests for the verifying the trainer functionality."""
import pytest

from rlgammon.trainer.step_trainer import StepTrainer


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
