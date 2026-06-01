"""Tests for the pure-float Janowski / gnubg cube-equity functions against published references."""

import pytest
import torch as th

from rlgammon.cube.cube_equity import (
    cash_point,
    cube_efficiency,
    cubeful_money_equity,
    cubeless_equity,
    take_point,
    w_and_l,
)
from rlgammon.cube.cube_errors.cube_errors import InvalidProbabilityVectorError
from rlgammon.cube.cube_types import CubeOwner, CubeState
from rlgammon.models.value_model import TDGammonNet

# The gammonless reference position: a 60% single-win chance and no gammons either way.
GAMMONLESS_PROBS = [0.6, 0.0, 0.0, 0.0, 0.0]
# A position with both gammon and backgammon mass on each side, for the identity test.
GAMMONFUL_PROBS = [0.55, 0.20, 0.05, 0.15, 0.03]
# Expected gammonless quantities.
GAMMONLESS_W = 1.0
GAMMONLESS_L = 1.0
GAMMONLESS_EQUITY = 0.2
# Reference take points for the gammonless game at three cube-life indices.
TAKE_POINT_X0 = 0.25
TAKE_POINT_X1 = 0.20
TAKE_POINT_X068 = 0.2137
# Janowski "Reno" example: average win 1.114, average loss 1.292 -> take point ~0.3292 at x = 0.
RENO_W = 1.114
RENO_L = 1.292
RENO_TAKE_POINT_X0 = 0.3292
# The default contact cube-life index.
DEFAULT_X = 0.68
# Race-interpolation reference points (pips -> efficiency).
RACE_PIPS_SHORT = 40.0
RACE_PIPS_LONG = 120.0
RACE_EFFICIENCY_SHORT = 0.6
RACE_EFFICIENCY_LONG = 0.7
# Tight and loose tolerances for the reference comparisons.
TIGHT_TOL = 1e-3
RENO_TOL = 2e-3


def test_cubeless_equity_matches_combine_equity() -> None:
    """Test that cubeless_equity equals the network's combine_equity for the same vector."""
    combined = float(TDGammonNet.combine_equity(th.tensor(GAMMONFUL_PROBS)))
    assert cubeless_equity(GAMMONFUL_PROBS) == pytest.approx(combined)


def test_cubeless_equity_matches_p_w_minus_l_identity() -> None:
    """Test the identity cubeless_equity == p*W - (1-p)*L on a gammonful position."""
    probs = GAMMONFUL_PROBS
    p = probs[0]
    avg_win, avg_lose = w_and_l(probs)
    assert cubeless_equity(probs) == pytest.approx(p * avg_win - (1.0 - p) * avg_lose)


def test_gammonless_w_and_l_and_equity() -> None:
    """Test that the gammonless position gives W = L = 1 and a cubeless equity of 0.2."""
    avg_win, avg_lose = w_and_l(GAMMONLESS_PROBS)
    assert avg_win == pytest.approx(GAMMONLESS_W)
    assert avg_lose == pytest.approx(GAMMONLESS_L)
    assert cubeless_equity(GAMMONLESS_PROBS) == pytest.approx(GAMMONLESS_EQUITY)


def test_gammonless_take_points() -> None:
    """Test the gammonless take points at x = 0, x = 1 and the default x = 0.68."""
    assert take_point(GAMMONLESS_W, GAMMONLESS_L, 0.0) == pytest.approx(TAKE_POINT_X0)
    assert take_point(GAMMONLESS_W, GAMMONLESS_L, 1.0) == pytest.approx(TAKE_POINT_X1)
    assert take_point(GAMMONLESS_W, GAMMONLESS_L, DEFAULT_X) == pytest.approx(TAKE_POINT_X068, abs=TIGHT_TOL)


def test_reno_take_point() -> None:
    """Test the Janowski Reno example take point W=1.114, L=1.292 -> ~0.3292 at x = 0."""
    assert take_point(RENO_W, RENO_L, 0.0) == pytest.approx(RENO_TAKE_POINT_X0, abs=RENO_TOL)


def test_cash_point_is_complement_of_take_point() -> None:
    """Test that the gammonless cash point equals 1 minus the take point (symmetry)."""
    take = take_point(GAMMONLESS_W, GAMMONLESS_L, DEFAULT_X)
    cash = cash_point(GAMMONLESS_W, GAMMONLESS_L, DEFAULT_X)
    assert cash == pytest.approx(1.0 - take)


def test_cube_efficiency_default_and_race_interpolation() -> None:
    """Test the default cube efficiency and the clamped race interpolation endpoints."""
    assert cube_efficiency() == pytest.approx(DEFAULT_X)
    assert cube_efficiency(RACE_PIPS_SHORT) == pytest.approx(RACE_EFFICIENCY_SHORT)
    assert cube_efficiency(RACE_PIPS_LONG) == pytest.approx(RACE_EFFICIENCY_LONG)
    assert cube_efficiency(0.0) == pytest.approx(RACE_EFFICIENCY_SHORT)
    assert cube_efficiency(1000.0) == pytest.approx(RACE_EFFICIENCY_LONG)


def test_cubeful_money_equity_centered_dead_cube_jacoby() -> None:
    """Test that a centred 1-cube under live Jacoby reduces to the cubeless money line 2p-1."""
    cube = CubeState(value=1, owner=CubeOwner.CENTERED, jacoby=True)
    probs = [0.6, 0.3, 0.1, 0.0, 0.0]
    assert cubeful_money_equity(probs, cube) == pytest.approx(2.0 * probs[0] - 1.0)


def test_cubeful_money_equity_owner_ordering() -> None:
    """Test that owning the cube is worth at least as much as the opponent owning it."""
    probs = [0.62, 0.15, 0.02, 0.10, 0.01]
    owned = cubeful_money_equity(probs, CubeState(value=2, owner=CubeOwner.ME))
    unavailable = cubeful_money_equity(probs, CubeState(value=2, owner=CubeOwner.OPP))
    assert owned > unavailable


def test_cubeful_money_equity_scales_with_cube_value() -> None:
    """Test that doubling the cube value doubles the cubeful money equity for a fixed owner."""
    probs = [0.62, 0.15, 0.02, 0.10, 0.01]
    one = cubeful_money_equity(probs, CubeState(value=1, owner=CubeOwner.ME))
    two = cubeful_money_equity(probs, CubeState(value=2, owner=CubeOwner.ME))
    assert two == pytest.approx(2.0 * one)


def test_invalid_probability_vector_raises() -> None:
    """Test that a probability vector of the wrong length raises InvalidProbabilityVectorError."""
    with pytest.raises(InvalidProbabilityVectorError):
        cubeless_equity([0.5, 0.1, 0.0])
