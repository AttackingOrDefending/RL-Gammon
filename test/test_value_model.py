"""Tests for the TD-Gammon value network and its equity head."""
import torch as th

from rlgammon.models.model_types import ValueHead
from rlgammon.models.value_model import N_EQUITY_COMPONENTS, N_INPUT_FEATURES, TDGammonNet

# Equity values expected from the cumulative-probability head for canonical outcomes.
GAMMON_WIN_EQUITY = 2.0
BACKGAMMON_WIN_EQUITY = 3.0
SINGLE_WIN_EQUITY = 1.0
GAMMON_LOSS_EQUITY = -2.0
BACKGAMMON_LOSS_EQUITY = -3.0
# Inclusive bounds for the equity-head output range.
EQUITY_LOWER_BOUND = -3.0
EQUITY_UPPER_BOUND = 3.0
# Inclusive bounds for the scalar-tanh head output range.
TANH_LOWER_BOUND = -1.0
TANH_UPPER_BOUND = 1.0
# Number of states stacked in the batched-combine test.
BATCH_SIZE = 3


def test_equity_forward_is_scalar_in_range() -> None:
    """Test that the equity-sigmoid forward pass returns a scalar within the (-3, 3) range."""
    net = TDGammonNet(value_head=ValueHead.EQUITY_SIGMOID)
    value = net([0.1] * N_INPUT_FEATURES)
    assert value.shape == ()
    assert EQUITY_LOWER_BOUND <= float(value) <= EQUITY_UPPER_BOUND


def test_combine_equity_gammon_win_exceeds_one() -> None:
    """Test that a gammon win combines to +2, proving the value can exceed +-1 (the bug-fix proof)."""
    raw = th.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    assert float(TDGammonNet.combine_equity(raw)) == GAMMON_WIN_EQUITY


def test_combine_equity_backgammon_win() -> None:
    """Test that a backgammon win combines to +3."""
    raw = th.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    assert float(TDGammonNet.combine_equity(raw)) == BACKGAMMON_WIN_EQUITY


def test_combine_equity_single_win() -> None:
    """Test that a single win combines to +1."""
    raw = th.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    assert float(TDGammonNet.combine_equity(raw)) == SINGLE_WIN_EQUITY


def test_combine_equity_gammon_loss() -> None:
    """Test that a gammon loss combines to -2."""
    raw = th.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    assert float(TDGammonNet.combine_equity(raw)) == GAMMON_LOSS_EQUITY


def test_combine_equity_backgammon_loss() -> None:
    """Test that a backgammon loss combines to -3."""
    raw = th.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    assert float(TDGammonNet.combine_equity(raw)) == BACKGAMMON_LOSS_EQUITY


def test_combine_equity_batched() -> None:
    """Test that combine_equity works on a batched (N, 5) tensor."""
    raw = th.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    combined = TDGammonNet.combine_equity(raw)
    assert combined.shape == (BATCH_SIZE,)
    assert combined.tolist() == [GAMMON_WIN_EQUITY, BACKGAMMON_WIN_EQUITY, SINGLE_WIN_EQUITY]


def test_scalar_tanh_forward_in_range() -> None:
    """Test that the scalar-tanh forward pass returns a scalar within the (-1, 1) range."""
    net = TDGammonNet(value_head=ValueHead.SCALAR_TANH)
    value = net([0.1] * N_INPUT_FEATURES)
    assert value.shape == ()
    assert TANH_LOWER_BOUND <= float(value) <= TANH_UPPER_BOUND


def test_raw_outputs_shapes() -> None:
    """Test that raw_outputs returns a 5-vector for the equity head and a 1-vector for the scalar head."""
    equity_net = TDGammonNet(value_head=ValueHead.EQUITY_SIGMOID)
    assert equity_net.raw_outputs([0.1] * N_INPUT_FEATURES).shape == (N_EQUITY_COMPONENTS,)

    scalar_net = TDGammonNet(value_head=ValueHead.SCALAR_TANH)
    assert scalar_net.raw_outputs([0.1] * N_INPUT_FEATURES).shape == (1,)
