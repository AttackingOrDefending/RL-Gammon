"""Tests for the per-move time budgeting policy of the TimeManager."""

import math

import pytest

from rlgammon.planning.time_manager import TimeManager

# A representative finite clock (seconds) used across the budgeting checks.
TIME_LEFT = 60.0
# A fixed reference timestamp so deadline arithmetic is deterministic.
NOW = 1000.0
# Tolerance for floating-point budget comparisons.
BUDGET_TOLERANCE = 1e-9


def test_infinite_time_is_unlimited() -> None:
    """Test that a non-finite clock yields an unlimited (None) budget and deadline."""
    manager = TimeManager()
    assert manager.budget_for_move(float("inf")) is None
    assert manager.deadline_for_move(float("inf"), now=NOW) is None
    assert manager.budget_for_move(math.nan) is None


def test_finite_budget_is_positive_and_bounded() -> None:
    """Test that a finite clock gives a budget that is positive, at least the floor and at most the clock."""
    min_move_time = 0.05
    manager = TimeManager(min_move_time=min_move_time)
    budget = manager.budget_for_move(TIME_LEFT)
    assert budget is not None
    assert budget >= min_move_time
    assert budget <= TIME_LEFT
    assert budget > 0.0


def test_budget_shrinks_as_game_progresses() -> None:
    """Test that as the game progresses (more moves played, less clock left) the budget shrinks."""
    manager = TimeManager()
    # Early game: full clock, no moves played yet.
    early = manager.budget_for_move(TIME_LEFT, move_number=0)
    # Late game: most of the clock spent and many moves already played.
    late = manager.budget_for_move(TIME_LEFT / 12, move_number=40)
    assert early is not None
    assert late is not None
    assert late <= early


def test_budget_shrinks_as_time_shrinks() -> None:
    """Test that a smaller remaining clock yields a smaller (or equal) budget at the same move."""
    manager = TimeManager()
    plenty = manager.budget_for_move(TIME_LEFT, move_number=0)
    scarce = manager.budget_for_move(TIME_LEFT / 10, move_number=0)
    assert plenty is not None
    assert scarce is not None
    assert scarce <= plenty


def test_budget_never_exceeds_time_left() -> None:
    """Test that even with a tiny clock the budget is clamped to at most the remaining time."""
    manager = TimeManager(min_move_time=10.0)
    tiny_clock = 1.0
    budget = manager.budget_for_move(tiny_clock, move_number=0)
    assert budget is not None
    assert budget <= tiny_clock


def test_deadline_is_now_plus_budget() -> None:
    """Test that the deadline equals the explicit reference time plus the allocated budget."""
    manager = TimeManager()
    budget = manager.budget_for_move(TIME_LEFT, move_number=4)
    deadline = manager.deadline_for_move(TIME_LEFT, move_number=4, now=NOW)
    assert budget is not None
    assert deadline is not None
    assert deadline == pytest.approx(NOW + budget, abs=BUDGET_TOLERANCE)
