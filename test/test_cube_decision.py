"""Tests for the gnubg compare-three-equities cube and take decisions."""

from rlgammon.cube.cube_equity import (
    CubeAction,
    TakeAction,
    double_decision,
    take_decision,
)
from rlgammon.cube.cube_types import CubeOwner, CubeState, GameMode, MatchContext

# A centred money 1-cube (Jacoby off, so gammons count).
CENTERED_MONEY_CUBE = CubeState(value=1, owner=CubeOwner.CENTERED)
# An even gammonless position (50% to win, no gammons).
EVEN_PROBS = [0.5, 0.0, 0.0, 0.0, 0.0]
# A strong gammonless position near the cash point.
STRONG_PROBS = [0.80, 0.0, 0.0, 0.0, 0.0]
# A marginal gammonless take position (22% to win) used at the cube-life extremes.
MARGINAL_TAKE_PROBS = [0.22, 0.0, 0.0, 0.0, 0.0]
# A position too good to double (heavy gammon mass with a high win chance).
TOO_GOOD_PROBS = [0.90, 0.80, 0.10, 0.0, 0.0]
# A first-to-7 Crawford match: the on-roll player trails 0-6.
CRAWFORD_MATCH_LENGTH = 7
CRAWFORD_OPP_SCORE = 6


def test_even_position_is_no_double() -> None:
    """Test that an even gammonless money position is not a double."""
    assert double_decision(EVEN_PROBS, CENTERED_MONEY_CUBE) == CubeAction.NO_DOUBLE


def test_strong_position_is_a_double() -> None:
    """Test that a strong gammonless money position offers a double (a pass/take label)."""
    action = double_decision(STRONG_PROBS, CENTERED_MONEY_CUBE)
    assert action in (CubeAction.DOUBLE_TAKE, CubeAction.DOUBLE_PASS)


def test_too_good_position_does_not_double() -> None:
    """Test that a position too good to double (large gammon mass) is labelled TOO_GOOD."""
    assert double_decision(TOO_GOOD_PROBS, CENTERED_MONEY_CUBE) == CubeAction.TOO_GOOD


def test_take_decision_depends_on_cube_efficiency() -> None:
    """Test that a 22% gammonless taker takes at x = 1 but passes at x = 0 (money)."""
    assert take_decision(MARGINAL_TAKE_PROBS, CENTERED_MONEY_CUBE, x=1.0) == TakeAction.TAKE
    assert take_decision(MARGINAL_TAKE_PROBS, CENTERED_MONEY_CUBE, x=0.0) == TakeAction.PASS


def test_take_decision_takes_strong_position() -> None:
    """Test that a clearly favoured taker takes the double."""
    assert take_decision([0.45, 0.0, 0.0, 0.0, 0.0], CENTERED_MONEY_CUBE) == TakeAction.TAKE


def test_cube_capped_at_max_does_not_double() -> None:
    """Test that a cube already at its ceiling cannot be doubled even from a strong position."""
    capped = CubeState(value=64, owner=CubeOwner.ME, max_cube=64)
    assert double_decision(STRONG_PROBS, capped) == CubeAction.NO_DOUBLE


def test_crawford_game_never_doubles() -> None:
    """Test that no double is offered in a Crawford game regardless of the win probability."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=0,
                       opp_score=CRAWFORD_OPP_SCORE)
    assert ctx.cube_dead_this_game is True
    for probs in (EVEN_PROBS, STRONG_PROBS, TOO_GOOD_PROBS):
        assert double_decision(probs, CENTERED_MONEY_CUBE, ctx) == CubeAction.NO_DOUBLE


def test_match_double_is_offered_when_ahead() -> None:
    """Test that a strong position in a non-Crawford match still offers a double."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=1, opp_score=1)
    assert ctx.cube_dead_this_game is False
    action = double_decision(STRONG_PROBS, CENTERED_MONEY_CUBE, ctx)
    assert action in (CubeAction.DOUBLE_TAKE, CubeAction.DOUBLE_PASS)


def test_match_take_at_double_match_point() -> None:
    """Test that an even taker takes when trailing far (a doubled cube can still win the match)."""
    ctx = MatchContext(mode=GameMode.MATCH, match_length=CRAWFORD_MATCH_LENGTH, my_score=0, opp_score=2)
    assert take_decision([0.5, 0.0, 0.0, 0.0, 0.0], CENTERED_MONEY_CUBE, ctx) == TakeAction.TAKE
