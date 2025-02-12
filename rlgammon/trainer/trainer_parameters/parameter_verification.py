from typing import Any


REQUIRED_PARAMETERS: list[tuple[str, type]] = [
    ("episodes", int)
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
