"""Frozen value types for the analytic doubling-cube / match-play layer.

The doubling cube is a pure analytic layer on top of the cubeless OpenSpiel game and the cubeless
value network; nothing in this module touches the game engine or torch. ``CubeState`` tracks the
stake and ownership of the cube; ``MatchContext`` tracks the match score and derives the away-counts,
the Crawford / post-Crawford state and whether the cube is dead for the current game.
"""

from dataclasses import dataclass
from enum import Enum

# The smallest legal cube value (a centred 1-cube at the start of a game).
INITIAL_CUBE_VALUE = 1
# Default doubling-cube ceiling (1, 2, 4, ..., 64) used in money and match play.
DEFAULT_MAX_CUBE = 64
# Away-count that triggers the Crawford game: exactly one side is one point from the match.
CRAWFORD_AWAY = 1


class CubeOwner(Enum):
    """Which side, relative to the on-roll player, may currently use the doubling cube."""

    CENTERED = "CENTERED"
    ME = "ME"
    OPP = "OPP"


class GameMode(Enum):
    """Whether equities are evaluated as money (per point) or as match-winning chance."""

    MONEY = "MONEY"
    MATCH = "MATCH"


@dataclass(frozen=True)
class CubeState:
    """
    Immutable state of the doubling cube from the on-roll player's perspective.

    :param value: the current cube value (stake multiplier: 1, 2, 4, ...)
    :param owner: who may currently double (centred, the on-roll player, or the opponent)
    :param jacoby: whether the Jacoby rule applies (gammons/backgammons score single until a double)
    :param beavers: whether beavers/raccoons are allowed (recorded only; money-play convention)
    :param max_cube: the maximum cube value the cube may be raised to
    """

    value: int = INITIAL_CUBE_VALUE
    owner: CubeOwner = CubeOwner.CENTERED
    jacoby: bool = False
    beavers: bool = False
    max_cube: int = DEFAULT_MAX_CUBE

    def can_double(self) -> bool:
        """
        Return whether the on-roll player may offer a double from this cube state.

        The on-roll player may double when the cube is centred or owned by them, and the resulting
        cube value would not exceed ``max_cube``.

        :return: whether a double is available to the on-roll player
        """
        below_ceiling = self.value * 2 <= self.max_cube
        return below_ceiling and self.owner in (CubeOwner.CENTERED, CubeOwner.ME)

    def after_double(self) -> "CubeState":
        """
        Return the cube state after the on-roll player doubles and the opponent takes.

        The value doubles and ownership passes to the opponent (the on-roll player gave up the cube).

        :return: the post-double cube state from the same (on-roll) perspective
        """
        return CubeState(value=self.value * 2, owner=CubeOwner.OPP, jacoby=self.jacoby,
                         beavers=self.beavers, max_cube=self.max_cube)

    def flip_perspective(self) -> "CubeState":
        """
        Return the same cube as seen from the opponent's point of view.

        Only ownership is perspective-dependent: ``ME`` and ``OPP`` swap while ``CENTERED`` is fixed.

        :return: the cube state from the opposite perspective
        """
        flipped = {CubeOwner.ME: CubeOwner.OPP, CubeOwner.OPP: CubeOwner.ME,
                   CubeOwner.CENTERED: CubeOwner.CENTERED}[self.owner]
        return CubeState(value=self.value, owner=flipped, jacoby=self.jacoby,
                         beavers=self.beavers, max_cube=self.max_cube)


@dataclass(frozen=True)
class MatchContext:
    """
    Immutable match score and rules, with derived away-counts and Crawford state.

    Scores are from the on-roll player's perspective (``my_score`` is the on-roll player's score).
    In :attr:`GameMode.MONEY` the match length and scores are ignored by the equity layer.

    :param mode: whether the game is scored as money or as match-winning chance
    :param match_length: the number of points needed to win the match (ignored in money play)
    :param my_score: the on-roll player's current match score
    :param opp_score: the opponent's current match score
    :param crawford_played: whether the Crawford game has already been played in this match
    """

    mode: GameMode = GameMode.MONEY
    match_length: int = 0
    my_score: int = 0
    opp_score: int = 0
    crawford_played: bool = False

    @property
    def my_away(self) -> int:
        """Return how many points the on-roll player still needs to win the match (>= 1)."""
        return max(self.match_length - self.my_score, CRAWFORD_AWAY)

    @property
    def opp_away(self) -> int:
        """Return how many points the opponent still needs to win the match (>= 1)."""
        return max(self.match_length - self.opp_score, CRAWFORD_AWAY)

    @property
    def is_crawford(self) -> bool:
        """
        Return whether this is the Crawford game (exactly one side is one point away, not yet played).

        :return: whether the current game is the Crawford game
        """
        if self.mode != GameMode.MATCH or self.crawford_played:
            return False
        return (self.my_away == CRAWFORD_AWAY) ^ (self.opp_away == CRAWFORD_AWAY)

    @property
    def is_post_crawford(self) -> bool:
        """
        Return whether this is a post-Crawford game (Crawford played and a side still one away).

        :return: whether the current game follows the Crawford game with the cube live again
        """
        if self.mode != GameMode.MATCH or not self.crawford_played:
            return False
        return CRAWFORD_AWAY in (self.my_away, self.opp_away)

    @property
    def cube_dead_this_game(self) -> bool:
        """
        Return whether the cube is dead for the whole game (the Crawford game: no doubling allowed).

        :return: whether doubling is forbidden for the entire current game
        """
        return self.is_crawford

    def flip_perspective(self) -> "MatchContext":
        """
        Return the same match seen from the opponent's point of view (scores swapped).

        :return: the match context from the opposite perspective
        """
        return MatchContext(mode=self.mode, match_length=self.match_length, my_score=self.opp_score,
                            opp_score=self.my_score, crawford_played=self.crawford_played)
