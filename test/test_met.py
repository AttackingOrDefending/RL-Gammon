"""Tests for the Woolsey-Heinrich match-equity table and its lookups."""

import pytest

from rlgammon.cube.cube_types import GameMode, MatchContext
from rlgammon.cube.met import WOOLSEY_HEINRICH

# Published reference cells of the Woolsey-Heinrich grid (my-away, opp-away).
MET_2_4 = 0.67
MET_1_2 = 0.68
MET_4_2 = 0.33
DIAGONAL_MWC = 0.5
# A first-to-7 match where the on-roll player trails 0-6 (the Crawford game).
CRAWFORD_MATCH_LENGTH = 7
CRAWFORD_OPP_SCORE = 6
CRAWFORD_MY_AWAY = 7
CRAWFORD_OPP_AWAY = 1


def test_reference_cells() -> None:
    """Test the published MWC cells MET[2][4], MET[1][2] and the anti-symmetric MET[4][2]."""
    assert WOOLSEY_HEINRICH.mwc_for_away(2, 4) == pytest.approx(MET_2_4)
    assert WOOLSEY_HEINRICH.mwc_for_away(1, 2) == pytest.approx(MET_1_2)
    assert WOOLSEY_HEINRICH.mwc_for_away(4, 2) == pytest.approx(MET_4_2)


def test_diagonal_is_even() -> None:
    """Test that equal away-counts always give a match-winning chance of 0.5."""
    for away in range(1, 8):
        assert WOOLSEY_HEINRICH.mwc_for_away(away, away) == pytest.approx(DIAGONAL_MWC)


def test_anti_symmetry() -> None:
    """Test the anti-symmetry MET[a][b] = 1 - MET[b][a] across the tabulated range."""
    for my_away in range(1, 8):
        for opp_away in range(1, 8):
            forward = WOOLSEY_HEINRICH.mwc_for_away(my_away, opp_away)
            backward = WOOLSEY_HEINRICH.mwc_for_away(opp_away, my_away)
            assert forward == pytest.approx(1.0 - backward)


def test_away_counts_are_clamped() -> None:
    """Test that away-counts beyond the table are clamped to the largest tabulated row/column."""
    assert WOOLSEY_HEINRICH.mwc_for_away(20, 3) == pytest.approx(WOOLSEY_HEINRICH.mwc_for_away(7, 3))
    assert WOOLSEY_HEINRICH.mwc_for_away(2, 99) == pytest.approx(WOOLSEY_HEINRICH.mwc_for_away(2, 7))


def test_mwc_from_match_context() -> None:
    """Test that the context-based lookup derives the correct away-counts and MWC."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=0,
                       opp_score=CRAWFORD_OPP_SCORE)
    assert ctx.my_away == CRAWFORD_MY_AWAY
    assert ctx.opp_away == CRAWFORD_OPP_AWAY
    assert WOOLSEY_HEINRICH.mwc(ctx) == pytest.approx(WOOLSEY_HEINRICH.mwc_for_away(7, 1))


def test_crawford_context_flags() -> None:
    """Test that a 0-6 score in a first-to-7 match is the Crawford game with a dead cube."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=0,
                       opp_score=CRAWFORD_OPP_SCORE)
    assert ctx.is_crawford is True
    assert ctx.cube_dead_this_game is True
    assert ctx.is_post_crawford is False


def test_post_crawford_context_flags() -> None:
    """Test that once Crawford is played and a side is one-away the cube is live again."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=0,
                       opp_score=CRAWFORD_OPP_SCORE, crawford_played=True)
    assert ctx.is_crawford is False
    assert ctx.is_post_crawford is True
    assert ctx.cube_dead_this_game is False


def test_money_context_is_never_crawford() -> None:
    """Test that a money-mode context is never a Crawford game regardless of the scores."""
    ctx = MatchContext(mode=GameMode.MONEY)
    assert ctx.is_crawford is False
    assert ctx.cube_dead_this_game is False
