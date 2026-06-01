"""The Woolsey-Heinrich match-equity table and lookups, as pure floats (no torch, no game engine).

A match-equity table (MET) gives the match-winning chance (MWC) for the player who is ``a`` points
away from winning the match against an opponent who is ``b`` points away. The grid is stored for the
lower triangle / diagonal and completed by the anti-symmetry ``MET[a][b] = 1 - MET[b][a]``; away-counts
beyond the table are clamped to the largest tabulated value (a standard, slightly conservative cut-off).
"""

from dataclasses import dataclass, field

from rlgammon.cube.cube_types import MatchContext

# The smallest away-count (a player one point from winning the match).
MIN_AWAY = 1
# The largest away-count tabulated in the Woolsey-Heinrich grid below.
MAX_AWAY = 7
# Match-winning chance on the diagonal (equal scores) and the trivial 1-vs-1 game.
EVEN_MWC = 0.5

# Pre-match-winning chance for the leader's row, i.e. MWC for ``a``-away vs ``b``-away with a <= b
# (the favoured side). The matrix is 1-indexed by away count and completed by anti-symmetry.
# Source: Kit Woolsey & Hal Heinrich match-equity table (post-Crawford rows folded into the grid).
_WOOLSEY_HEINRICH_GRID: tuple[tuple[float, ...], ...] = (
    (0.500, 0.680, 0.750, 0.815, 0.845, 0.895, 0.915),
    (0.320, 0.500, 0.600, 0.670, 0.750, 0.800, 0.850),
    (0.250, 0.400, 0.500, 0.575, 0.650, 0.720, 0.780),
    (0.185, 0.330, 0.425, 0.500, 0.575, 0.640, 0.700),
    (0.155, 0.250, 0.350, 0.425, 0.500, 0.570, 0.630),
    (0.105, 0.200, 0.280, 0.360, 0.430, 0.500, 0.560),
    (0.085, 0.150, 0.220, 0.300, 0.370, 0.440, 0.500),
)


@dataclass(frozen=True)
class MET:
    """
    An immutable match-equity table holding a square grid of match-winning chances.

    :param grid: rows of MWC values, 0-indexed (row ``i``, column ``j`` is ``(i+1)``-away vs ``(j+1)``-away)
    """

    grid: tuple[tuple[float, ...], ...] = field(default=_WOOLSEY_HEINRICH_GRID)

    def mwc_for_away(self, my_away: int, opp_away: int) -> float:
        """
        Return the on-roll player's match-winning chance for the given away-counts.

        Away-counts are clamped to ``[1, len(grid)]``; the largest tabulated row/column is reused
        beyond the table. Equal away-counts return exactly ``0.5``.

        :param my_away: how many points the on-roll player needs to win the match (>= 1)
        :param opp_away: how many points the opponent needs to win the match (>= 1)
        :return: the on-roll player's match-winning chance in ``[0, 1]``
        """
        size = len(self.grid)
        my_index = min(max(my_away, MIN_AWAY), size) - 1
        opp_index = min(max(opp_away, MIN_AWAY), size) - 1
        if my_index == opp_index:
            return EVEN_MWC
        return self.grid[my_index][opp_index]

    def mwc(self, match_ctx: MatchContext) -> float:
        """
        Return the on-roll player's pre-game match-winning chance for a match context.

        :param match_ctx: the match score and rules (the on-roll player's away-counts are derived)
        :return: the on-roll player's match-winning chance in ``[0, 1]``
        """
        return self.mwc_for_away(match_ctx.my_away, match_ctx.opp_away)


# Module-level default match-equity table used throughout the cube layer.
WOOLSEY_HEINRICH = MET()
