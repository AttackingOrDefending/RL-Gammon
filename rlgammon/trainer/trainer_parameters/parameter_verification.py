from typing import Any

from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.exploration.exploration_types import PossibleExploration

REQUIRED_PARAMETERS: list[tuple[str, type]] = [
    ("episodes", int),
    ("batch_size", int),
    ("buffer", PossibleBuffers),
    ("exploration", PossibleExploration),
    ("decay", float)
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
            if type(value) is not parameter_type:
                return False
        except KeyError:
            return False
    return True
