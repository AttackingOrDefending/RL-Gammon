"""Type aliases for exploration algorithms."""

from enum import Enum


class PossibleExploration(Enum):
    """Enumeration of possible exploration types."""

    EPSILON_GREEDY = "EG"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> 'PossibleExploration':
        """
        Convert string, found e.g. in JSON parameters to a PossibleExploration enum.

        :return: the corresponding enum, if none found, return null
        """
        if string_to_convert == PossibleExploration.EPSILON_GREEDY.value:
            return PossibleExploration.EPSILON_GREEDY
        return None  # type: ignore
