"""Define different types related to testing objects."""

from enum import Enum


class PossibleTesting(Enum):
    """Enumeration of possible testing types."""

    RANDOM = "RND"
    GNU = "GNU"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleTesting":
        """
        Convert string, found e.g. in JSON parameters to a PossibleTesters enum.

        :return: the corresponding enum, if none found, return null
        """
        if string_to_convert == PossibleTesting.RANDOM.value:
            return PossibleTesting.RANDOM

        match string_to_convert:
            case "RND":
                return PossibleTesting.RANDOM
            case "GNU":
                return PossibleTesting.GNU
            case _:
                msg = f"'{string_to_convert}' is not a valid testing type string. Try 'RND' or 'GNU'."
                raise ValueError(msg)
