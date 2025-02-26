"""Type aliases for exploration algorithms."""

from enum import Enum


class PossibleExploration(Enum):
    """Enumeration of possible exploration types."""

    EPSILON_GREEDY = "EG"

    @staticmethod
    def get_enum_from_string(string_to_convert: str):
        """
        TODO

        :return:
        """
        if string_to_convert == PossibleExploration.EPSILON_GREEDY.value:
            return PossibleExploration.EPSILON_GREEDY
        return None
