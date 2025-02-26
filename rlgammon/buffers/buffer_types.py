"""Type aliases for buffers."""

from typing import Any
from enum import Enum

from numpy.typing import NDArray

BufferBatch = dict[str, NDArray[Any]]


class PossibleBuffers(Enum):
    """Enumeration of possible buffer types."""

    UNIFORM = "U"

    @staticmethod
    def get_enum_from_string(string_to_convert: str):
        """
        TODO

        :return:
        """
        if string_to_convert == PossibleBuffers.UNIFORM.value:
            return PossibleBuffers.UNIFORM
        return None
