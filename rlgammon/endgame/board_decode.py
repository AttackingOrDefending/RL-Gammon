"""Decode an OpenSpiel backgammon observation tensor into an explicit per-side checker layout.

Phase detection and the bear-off specialist both need each side's checker positions (the 24 points,
the bar and the off tray), which the value network never sees explicitly. This module recovers them
from ``state.observation_tensor(perspective)`` alone, so the rest of the package depends only on the
``GameState`` protocol and stays unit-testable without OpenSpiel installed.

Observation-tensor layout (verified empirically against ``backgammon(scoring_type=full_scoring)``;
length 200 from ``perspective``'s point of view):

* floats ``[0:96]``   -- ``perspective``'s 24 points, four floats per point;
* floats ``[96:192]`` -- the opponent's 24 points, same four-float scheme;
* float  ``192``      -- ``perspective``'s checkers on the bar (raw count);
* float  ``193``      -- ``perspective``'s checkers borne off (raw count);
* float  ``195``      -- the opponent's checkers on the bar;
* float  ``196``      -- the opponent's checkers borne off;
  (the remaining tail floats encode internal flags and the two dice and are ignored here).

Each four-float point group ``(a, b, c, d)`` encodes a checker count ``n`` as a unary prefix with an
overflow tail: ``a = [n == 1]``, ``b = [n == 2]``, ``c = [n == 3]`` and ``d = n - 3`` when ``n > 3``
(``d == 0`` otherwise), so ``n`` is recovered exactly.

``observation_tensor(perspective)`` always lists ``perspective``'s own checkers in the first block
and the opponent's in the second, but each block's *direction is tied to the physical player that
owns it*, because the two players travel opposite ways round the board: WHITE (player 0) is encoded
with tensor point index ``i`` (``0..23``) at pip distance ``24 - i`` (index 23 = WHITE's 1-point),
whereas BLACK (player 1) is encoded with index ``i`` at pip distance ``i + 1`` (index 0 = BLACK's
1-point). Consequently the *same* physical player decodes to the *same* layout whether read as the
own block from its own perspective or as the opponent block from the other perspective.

To give every consumer one uniform frame the decoder reverses BLACK's block, so every returned
:class:`SideLayout` obeys **point index ``i`` is at pip distance ``24 - i``** in that side's own
direction: index ``23`` is its 1-point, index ``18`` its 6-point, the home board is indices
``18..23``, and a checker on the bar is ``25`` pips from off. This reproduces the standard 167-pip
opening count for each side and makes the decoded phase perspective-independent.
"""

from dataclasses import dataclass

from rlgammon.endgame.endgame_errors.endgame_errors import ObservationTensorLengthError
from rlgammon.endgame.endgame_types import CHECKERS_PER_SIDE, HOME_BOARD_SIZE
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.rlgammon_types import BLACK, WHITE

# Expected length of the backgammon observation tensor (198 board features + 2 dice).
OBSERVATION_TENSOR_LENGTH = 200
# Number of board points.
NUM_POINTS = 24
# Floats used to encode the checker count on a single point.
FLOATS_PER_POINT = 4
# Offset (in floats) of the opponent's point block within the observation tensor.
OPPONENT_BLOCK_OFFSET = NUM_POINTS * FLOATS_PER_POINT
# Tail-float offsets for the bar/off counters.
OWN_BAR_OFFSET = 192
OWN_OFF_OFFSET = 193
OPPONENT_BAR_OFFSET = 195
OPPONENT_OFF_OFFSET = 196
# Pip distance of a checker sitting on the bar.
BAR_PIP_DISTANCE = 25
# Lowest pip distance still inside the home board (the 6-point); 1..6 are the home points.
HOME_MAX_PIP = HOME_BOARD_SIZE
# A count-encoding float is treated as "set" above this threshold (the unary entries are 0 or 1).
ENCODING_HALF = 0.5
# A float is treated as a positive count above this threshold.
ENCODING_EPSILON = 1e-9


def _decode_point_count(group: list[float]) -> int:
    """
    Decode the checker count on a single point from its four-float observation group.

    :param group: the four floats ``(a, b, c, d)`` encoding the point
    :return: the number of checkers on the point (``0`` if empty)
    """
    a, b, c, d = group
    if a > ENCODING_HALF:
        return 1
    if b > ENCODING_HALF:
        return 2
    if c > ENCODING_HALF:
        return 3
    if d > ENCODING_EPSILON:
        return round(d) + 3
    return 0


@dataclass(frozen=True)
class SideLayout:
    """The full checker layout of one side, expressed in that side's own travelling direction.

    ``points`` has length 24; ``points[i]`` is the number of this side's checkers at pip distance
    ``24 - i`` from bearing off (index 23 = the 1-point, index 18 = the 6-point, index 0 = the
    24-point). ``bar`` and ``off`` are the checkers on the bar and already borne off.
    """

    points: tuple[int, ...]
    bar: int
    off: int

    def pip_count(self) -> int:
        """
        Return this side's pip count (sum of pip distances over all checkers, bar counted as 25).

        :return: the total pips this side still has to travel to bear every checker off
        """
        board_pips = sum(count * (NUM_POINTS - index) for index, count in enumerate(self.points))
        return board_pips + self.bar * BAR_PIP_DISTANCE

    def checkers_outside_home(self) -> int:
        """
        Return how many of this side's checkers are not yet in its home board (bar counts as outside).

        :return: the number of checkers on the bar or on points 7..24 (pip distance ``> 6``)
        """
        outside_points = sum(
            count for index, count in enumerate(self.points) if (NUM_POINTS - index) > HOME_MAX_PIP
        )
        return outside_points + self.bar

    def all_home(self) -> bool:
        """
        Return whether every one of this side's checkers is in its home board or already off.

        :return: ``True`` iff no checker is on the bar or on a point at pip distance ``> 6``
        """
        return self.checkers_outside_home() == 0

    def rearmost_pip(self) -> int:
        """
        Return the pip distance of this side's rearmost checker (its furthest-from-off checker).

        A checker on the bar (pip distance 25) dominates; with no checkers left at all the rearmost
        pip is ``0``.

        :return: the largest pip distance occupied by any of this side's checkers (``0`` if none)
        """
        if self.bar > 0:
            return BAR_PIP_DISTANCE
        for index, count in enumerate(self.points):
            if count > 0:
                return NUM_POINTS - index
        return 0

    def home_config(self) -> tuple[int, ...]:
        """
        Return this side's home board as six counts ordered from the 6-point down to the 1-point.

        This is only meaningful when :meth:`all_home` holds; callers must check the phase first.
        Index 0 is the 6-point (pip distance 6) and index 5 is the 1-point (pip distance 1).

        :return: a length-6 tuple of home-point checker counts (6-point first)
        """
        # Home points are indices 18..23 (pip 6..1); reverse so the 6-point comes first.
        return tuple(self.points[NUM_POINTS - pip] for pip in range(HOME_MAX_PIP, 0, -1))


@dataclass(frozen=True)
class BoardLayout:
    """The decoded layout of both sides, each in its own travelling direction.

    ``mover`` is the side whose perspective the observation tensor was taken from; ``opponent`` is
    the other side. Both are :class:`SideLayout` instances indexed in their *own* direction, so the
    same pip / home conventions apply to each.
    """

    mover: SideLayout
    opponent: SideLayout


def _decode_side(values: list[float], block_offset: int, bar_offset: int, off_offset: int,
                 physical_player: int) -> SideLayout:
    """
    Decode one physical player's :class:`SideLayout` from its point block and bar/off counters.

    The tensor encodes each block in its physical owner's own travelling direction: WHITE (player 0)
    with tensor point index ``i`` at pip distance ``24 - i`` and BLACK (player 1) with index ``i`` at
    pip distance ``i + 1`` (the players go opposite ways). To return one uniform "index ``i`` = pip
    distance ``24 - i``" layout for both, BLACK's block is reversed (``i -> 23 - i``); WHITE's is kept
    as-is. The decode therefore yields the same layout for a given physical player from either
    perspective's tensor.

    :param values: the full observation-tensor float list
    :param block_offset: the float offset of this player's 24-point block
    :param bar_offset: the float offset of this player's bar counter
    :param off_offset: the float offset of this player's off counter
    :param physical_player: the physical player owning the block (WHITE=0, BLACK=1)
    :return: the decoded side layout, indexed so that point index ``i`` is at pip distance ``24 - i``
    """
    raw_points = [
        _decode_point_count(values[block_offset + point * FLOATS_PER_POINT: block_offset + (point + 1) * FLOATS_PER_POINT])
        for point in range(NUM_POINTS)
    ]
    points = tuple(reversed(raw_points)) if physical_player == BLACK else tuple(raw_points)
    return SideLayout(points=points, bar=round(values[bar_offset]), off=round(values[off_offset]))


def decode_board(state: GameState, perspective: int) -> BoardLayout:
    """
    Decode the board into per-side checker layouts from ``perspective``'s observation tensor.

    The returned :attr:`BoardLayout.mover` is ``perspective``'s own layout regardless of whose turn
    it is, so callers can reason about a fixed side; :attr:`BoardLayout.opponent` is the other side
    (physical player ``1 - perspective``). Both layouts are normalised to the uniform pip convention
    of :class:`SideLayout` (index ``i`` = pip distance ``24 - i``), so the decode of a given physical
    player is identical from either perspective.

    :param state: the game state to decode (must expose a length-200 backgammon observation tensor)
    :param perspective: the player whose perspective to decode (WHITE=0, BLACK=1)
    :return: the decoded layout of both sides
    :raises ObservationTensorLengthError: if the observation tensor is not length 200
    """
    values = state.observation_tensor(perspective)
    if len(values) != OBSERVATION_TENSOR_LENGTH:
        raise ObservationTensorLengthError
    mover = _decode_side(values, 0, OWN_BAR_OFFSET, OWN_OFF_OFFSET, perspective)
    opponent = _decode_side(values, OPPONENT_BLOCK_OFFSET, OPPONENT_BAR_OFFSET, OPPONENT_OFF_OFFSET, 1 - perspective)
    return BoardLayout(mover=mover, opponent=opponent)


def side_layout_for(layout: BoardLayout, perspective: int, decoded_from: int = WHITE) -> SideLayout:
    """
    Return the :class:`SideLayout` of ``perspective`` from a layout decoded for ``decoded_from``.

    :param layout: a board layout produced by :func:`decode_board` for player ``decoded_from``
    :param perspective: the side whose layout to return (WHITE=0, BLACK=1)
    :param decoded_from: the perspective the layout was decoded from (defaults to WHITE)
    :return: ``perspective``'s side layout (``mover`` if it matches ``decoded_from``, else ``opponent``)
    """
    return layout.mover if perspective == decoded_from else layout.opponent


def _validate_checker_total(side: SideLayout) -> bool:
    """
    Return whether a side layout accounts for exactly the full complement of checkers.

    :param side: the side layout to check
    :return: ``True`` iff points, bar and off sum to ``CHECKERS_PER_SIDE``
    """
    return sum(side.points) + side.bar + side.off == CHECKERS_PER_SIDE
