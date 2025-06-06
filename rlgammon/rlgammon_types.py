"""Type aliases for rl-gammon package."""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

WHITE = 0
BLACK = 1

MIN_DICE = 1
MAX_DICE = 6

# GNU
ActionGNU = tuple[tuple[int, int], ...]
ActionSetGNU = list[int]
StateGNU = NDArray[np.float32]

# OPEN SPIEL
Features = list[float]

MovePart = tuple[int, int]
MoveDict = dict[int, set[MovePart]]
Move = list[MovePart]
MoveList = list[tuple[int, MovePart]]

Color = tuple[int, int, int]

Board = NDArray[np.int8]
Bar = NDArray[np.int8]
Off = NDArray[np.int8]
Input = NDArray[np.int8] | NDArray[np.float16]


class Orientation(Enum):
    """Orientation of the board."""

    TOP = "top"
    BOTTOM = "bottom"


class CheckerColor(Enum):
    """CheckerColor of the checkers."""

    WHITE = "W"
    BLACK = "B"
    NONE = ""
