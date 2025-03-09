"""Test for verifying the functionality of checking the validity of parameters."""

import json
from pathlib import Path
from typing import Any

import pytest

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid


@pytest.fixture
def parameters(request) -> dict[str, Any]:  # type: ignore[no-untyped-def] # noqa: ANN001
    """
    Setup function to load json parameters.

    :param request: parameters to the fixture
    :return: loaded json parameters as a dictionary
    """
    json_parameters_name = request.param
    parameter_file_path = Path(__file__).parent.parent
    parameter_file_path = parameter_file_path.joinpath("rlgammon/trainer/trainer_parameters/parameters/test_parameters/")
    path = parameter_file_path.joinpath(json_parameters_name)
    with path.open() as json_parameters:
        return json.load(json_parameters)  # type: ignore[no-any-return]


@pytest.mark.parametrize("parameters", ["valid_test_parameters.json"], indirect=True)
def test_valid_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that valid parameters are accepted.

    :param parameters: json parameters received from the fixture
    """
    assert are_parameters_valid(parameters)


@pytest.mark.parametrize("parameters", ["invalid_key_test_parameters.json"], indirect=True)
def test_invalid_key_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that invalid parameters, with an invalid key are rejected.

    :param parameters: json parameters received from the fixture
    """
    assert not are_parameters_valid(parameters)


@pytest.mark.parametrize("parameters", ["invalid_type_test_parameters.json"], indirect=True)
def test_invalid_type_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that invalid parameters, with an invalid type are rejected.

    :param parameters: json parameters received from the fixture
    """
    assert not are_parameters_valid(parameters)


@pytest.mark.parametrize("parameters", ["invalid_buffer_type_test_parameters.json"], indirect=True)
def test_invalid_buffer_type_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that invalid parameters, with a wrong buffer type are rejected.

    :param parameters: json parameters received from the fixture
    """
    assert not are_parameters_valid(parameters)


@pytest.mark.parametrize("parameters", ["invalid_exploration_type_test_parameters.json"], indirect=True)
def test_invalid_exploration_type_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that invalid parameters, with a wrong exploration type are rejected.

    :param parameters: json parameters received from the fixture
    """
    assert not are_parameters_valid(parameters)


@pytest.mark.parametrize("parameters", ["invalid_testing_type_test_parameters.json"], indirect=True)
def test_invalid_testing_type_parameter_verification(parameters: dict[str, Any]) -> None:
    """
    Test that invalid parameters, with a wrong testing type are rejected.

    :param parameters: json parameters received from the fixture
    """
    assert not are_parameters_valid(parameters)
