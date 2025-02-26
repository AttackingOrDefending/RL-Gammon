from typing import Any

from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.exploration.exploration_types import PossibleExploration

REQUIRED_PARAMETERS: list[tuple[str, type]] = [
    ("episodes", int),
    ("batch_size", int),
    ("buffer", PossibleBuffers),
    ("buffer_capacity", int),
    ("exploration", PossibleExploration),
    ("decay", float),
    ("start_epsilon", float),
    ("end_epsilon", float),
    ("update_decay", float),
    ("steps_per_update", int)
]


def are_parameters_valid(parameters: dict[str, Any]) -> bool:
    """
    TODO

    :param parameters:
    :return:
    """

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

            if type(value) is not parameter_type:
                return False
        except KeyError:
            return False
    return True
