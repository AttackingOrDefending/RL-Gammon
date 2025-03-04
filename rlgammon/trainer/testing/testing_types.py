"""TODO."""

from enum import Enum


class PossibleTesting(Enum):
    """TODO."""

    RANDOM = "RND"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleTesting":
        """
        Convert string, found e.g. in JSON parameters to a PossibleTesters enum.

        :return: the corresponding enum, if none found, return null
        """
        if string_to_convert == PossibleTesting.RANDOM.value:
            return PossibleTesting.RANDOM
        return None  # type: ignore[return-value]
