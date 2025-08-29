"""Type aliases for exploration algorithms."""

from enum import Enum


class PossibleExploration(Enum):
    """Enumeration of possible exploration types."""

    EPSILON_GREEDY = "EG"
    NO_EXPLORATION = "NO"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleExploration":
        """
        Convert string, found e.g. in JSON parameters to a PossibleExploration enum.

        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "EG":
                return PossibleExploration.EPSILON_GREEDY
            case "NO":
                return PossibleExploration.NO_EXPLORATION
            case _:
                msg = f"'{string_to_convert}' is not a valid exploration type string. Try 'EG' or 'NO'."
                raise ValueError(msg)