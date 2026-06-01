"""Type aliases and enums for the game engine boundary."""

from enum import Enum

# A player id. WHITE=0, BLACK=1 (see rlgammon.rlgammon_types). Negative values are
# OpenSpiel sentinels for chance (-1) and terminal (-4) nodes.
Player = int
# An action id in the unified OpenSpiel action space (chance outcomes and player moves share it).
Action = int
# A single chance outcome: the action id of the dice roll and its probability.
ChanceOutcome = tuple[int, float]
# The full observation tensor returned by the engine (198 board features + 2 dice for backgammon).
ObsTensor = list[float]
# The board-only feature slice fed to the value network.
Features = list[float]
# The per-player signed returns at a terminal state.
Returns = list[float]


class PossibleEngine(Enum):
    """Enumeration of possible game engine backends."""

    OPEN_SPIEL = "OS"
    MOCK = "MOCK"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleEngine":
        """
        Convert a string, found e.g. in JSON parameters, to a PossibleEngine enum.

        :param string_to_convert: the string value to convert
        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "OS":
                return PossibleEngine.OPEN_SPIEL
            case "MOCK":
                return PossibleEngine.MOCK
            case _:
                return None  # type: ignore[return-value]
