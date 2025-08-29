from enum import Enum


class TrainerType(Enum):
    """TODO."""

    STEP_TRAINER = "ST"
    ITERATION_TRAINER = "IT"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "TrainerType":
        """
        Convert string, found e.g. in JSON parameters to a PossibleExploration enum.

        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "ST":
                return TrainerType.STEP_TRAINER
            case "IT":
                return TrainerType.ITERATION_TRAINER
            case _:
                msg = f"'{string_to_convert}' is not a valid trainer type string. Try 'ST' or 'IT'."
                raise ValueError(msg)
