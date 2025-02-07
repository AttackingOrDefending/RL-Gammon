"""Type aliases for rl-gammon package."""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

MovePart = tuple[int, int]
MoveDict = dict[int, set[MovePart]]
Move = list[MovePart]

Board = NDArray[np.int8]
Bar = NDArray[np.int8]
Off = NDArray[np.int8]


class Orientation(Enum):
    """Orientation of the board."""

    TOP = "top"
    BOTTOM = "bottom"


class Color(Enum):
    """Color of the checkers."""

    WHITE = "W"
    BLACK = "B"
    NONE = ""
