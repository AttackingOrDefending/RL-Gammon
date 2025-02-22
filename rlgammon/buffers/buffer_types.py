"""Type aliases for buffers."""

from typing import Any
from enum import Enum

from numpy.typing import NDArray

BufferBatch = dict[str, NDArray[Any]]


class PossibleBuffers(Enum):
    """TODO"""

    UNIFORM = "U"
