"""Type aliases and the game-phase enum for the endgame package."""

from enum import Enum

# Number of points in a backgammon home board (the six points a side bears off from).
HOME_BOARD_SIZE = 6
# Total checkers per side.
CHECKERS_PER_SIDE = 15

# A home-board configuration for ONE side: a length-6 tuple giving the number of checkers on each
# home point, ordered from the 6-point (index 0) down to the 1-point (index 5). The number of
# checkers already borne off is implied as ``CHECKERS_PER_SIDE - sum(config)`` and never stored, so
# the configuration is a complete, hashable key for the one-sided bear-off distribution.
HomeConfig = tuple[int, int, int, int, int, int]

# A pip-indexed checker layout for ONE side: ``points[d]`` is the number of that side's checkers at
# pip distance ``d`` from bearing off, for ``d`` in ``1..24`` (index 0 unused / always 0). Checkers
# on the bar are folded in at pip distance 25 by the decoder before this is built where relevant.
PipCounts = tuple[int, ...]


class Phase(Enum):
    """The coarse phase of a backgammon position, used to route leaf evaluation.

    * ``CONTACT``: the two sides can still hit each other (some checker of one side is behind some
      checker of the other), so tactical neural-net evaluation is needed.
    * ``RACE``: no contact is possible, but at least one side still has a checker outside its home
      board, so it is a pure (possibly long) race.
    * ``BEAROFF``: every checker of BOTH sides is in its home board (or already off), so the exact
      bear-off database applies directly.
    """

    CONTACT = "CONTACT"
    RACE = "RACE"
    BEAROFF = "BEAROFF"
