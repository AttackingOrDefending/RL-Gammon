"""Classify a backgammon position as CONTACT, RACE or BEAROFF from the decoded board.

The phase decides which leaf evaluator a composite evaluator should trust (see
:mod:`rlgammon.endgame.composite_evaluator`): once the two sides can no longer touch, the outcome is
a pure race and an exact specialist is sharper than any neural net.

Phase rule (computed on the decoded :class:`~rlgammon.endgame.board_decode.BoardLayout`):

* **CONTACT** -- the two sides can still hit each other. Place both sides on one 24-point board: a
  mover checker at pip distance ``d`` sits on absolute point ``d`` (mover bears off past point 0),
  while an opponent checker at the opponent's pip distance ``e`` sits on absolute point ``25 - e``
  (the opponent travels the other way). The mover's rearmost checker is at absolute point
  ``mover_rearmost`` and the opponent's at absolute point ``25 - opponent_rearmost``; they can still
  meet -- a checker of one side is at or behind a checker of the other -- exactly when
  ``mover_rearmost >= 25 - opponent_rearmost``, i.e. ``mover_rearmost + opponent_rearmost >= 25``.
  A checker on the bar (pip distance 25) always forces contact under this test, which is correct: it
  re-enters in the opponent's home board, behind opposing checkers.
* **BEAROFF** -- not contact, and *both* sides have every checker in their home board (or off), so
  the exact two-sided bear-off database applies directly.
* **RACE** -- not contact and not bear-off: the sides are disengaged but at least one still has a
  checker outside its home board, so it is a (possibly long) pure race.
"""

from rlgammon.endgame.board_decode import BoardLayout, SideLayout, decode_board
from rlgammon.endgame.endgame_types import Phase
from rlgammon.game.backgammon_protocol import GameState

# A mover checker at pip ``d`` and an opponent checker at pip ``e`` share the 24-point board through
# the relation (absolute mover point) = d and (absolute opponent point) = 25 - e; contact is
# possible when those rearmost points cross, i.e. d + e >= this threshold.
CONTACT_PIP_SUM_THRESHOLD = 25


def _sides_can_contact(mover: SideLayout, opponent: SideLayout) -> bool:
    """
    Return whether the two sides can still hit each other (some checker is behind an opposing one).

    :param mover: one side's layout (in its own travelling direction)
    :param opponent: the other side's layout (in its own travelling direction)
    :return: ``True`` iff the sides' occupied ranges overlap on a shared 24-point board
    """
    mover_rearmost = mover.rearmost_pip()
    opponent_rearmost = opponent.rearmost_pip()
    # If either side has no checkers left on the board (all off), no contact is possible.
    if mover_rearmost == 0 or opponent_rearmost == 0:
        return False
    return mover_rearmost + opponent_rearmost >= CONTACT_PIP_SUM_THRESHOLD


def detect_phase_from_layout(layout: BoardLayout) -> Phase:
    """
    Classify a decoded board layout as CONTACT, RACE or BEAROFF.

    :param layout: the decoded layout of both sides (see :func:`decode_board`)
    :return: the phase of the position
    """
    if _sides_can_contact(layout.mover, layout.opponent):
        return Phase.CONTACT
    if layout.mover.all_home() and layout.opponent.all_home():
        return Phase.BEAROFF
    return Phase.RACE


def detect_phase(state: GameState, perspective: int) -> Phase:
    """
    Classify ``state`` as CONTACT, RACE or BEAROFF (the result is perspective-independent).

    The classification depends only on the two sides' relative positions, so passing either player
    as ``perspective`` yields the same phase; ``perspective`` only selects whose observation tensor
    is decoded.

    :param state: the game state to classify (must expose a backgammon observation tensor)
    :param perspective: the player whose observation tensor to decode (WHITE=0, BLACK=1)
    :return: the phase of the position
    :raises ObservationTensorLengthError: if the observation tensor is not length 200
    """
    return detect_phase_from_layout(decode_board(state, perspective))
