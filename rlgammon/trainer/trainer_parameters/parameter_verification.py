"""Verify parameters passed to the trainer."""

from typing import Any

from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.trainer.testing.testing_types import PossibleTesting

REQUIRED_PARAMETERS: list[tuple[str, type]] = [
    ("episodes", int),
    ("save_every", int),
    ("testing_type", PossibleTesting),
    ("episodes_in_test", int),
    ("episodes_per_test", int),
    ("batch_size", int),
    ("buffer", PossibleBuffers),
    ("buffer_capacity", int),
    ("exploration", PossibleExploration),
    ("decay", float),
    ("start_epsilon", float),
    ("end_epsilon", float),
    ("update_decay", float),
    ("steps_per_update", int),
    ("load_logger", bool),
    ("logger_name", str),
    ("save_progress", bool),
    ("iterations", int)
]


def are_parameters_valid(parameters: dict[str, Any]) -> bool:
    """
    Check if the loaded parameters are valid. To be valid,
    they must contain all data points from the 'REQUIRED_PARAMETERS' list, with the correct data types.

    :param parameters:
    :return: boolean indicating whether parameters are valid, true for valid, else false
    """
    if len(parameters) != len(REQUIRED_PARAMETERS):
        return False

    for parameter in REQUIRED_PARAMETERS:
        parameter_key = parameter[0]
        parameter_type = parameter[1]
        try:
            value = parameters[parameter_key]

            # Convert from JSON string to Enum, if applicable
            if parameter_type == PossibleBuffers:
                value = PossibleBuffers.get_enum_from_string(value)
                parameters[parameter_key] = value
            elif parameter_type == PossibleExploration:
                value = PossibleExploration.get_enum_from_string(value)
                parameters[parameter_key] = value
            elif parameter_type == PossibleTesting:
                value = PossibleTesting.get_enum_from_string(value)
                parameters[parameter_key] = value

            if type(value) is not parameter_type:
                return False
        except KeyError:
            return False
    return True
