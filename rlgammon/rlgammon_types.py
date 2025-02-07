"""Type aliases for rl-gammon package."""

import numpy as np
from numpy.typing import NDArray

MovePart = tuple[int, int]
MoveDict = dict[int, set[MovePart]]
Move = list[MovePart]

Board = NDArray[np.int8]
Bar = NDArray[np.int8]
Off = NDArray[np.int8]
