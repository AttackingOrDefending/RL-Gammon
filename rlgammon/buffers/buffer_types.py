"""Type aliases for buffers."""

from enum import Enum
from typing import Any

from numpy.typing import NDArray

BufferBatch = dict[str, NDArray[Any]]


class PossibleBuffers(Enum):
    """Enumeration of possible buffer types."""

    UNIFORM = "U"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "PossibleBuffers":
        """
        Convert string, found e.g. in JSON parameters to a PossibleBuffers enum.

        :return: the corresponding enum, if none found, return null
        """
        if string_to_convert == PossibleBuffers.UNIFORM.value:
            return PossibleBuffers.UNIFORM
        return None  # type: ignore[return-value]
