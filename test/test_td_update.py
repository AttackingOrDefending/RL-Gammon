"""Tests for the corrected, undiscounted TD(lambda) weight update."""
from typing import cast

import torch as th
from torch import nn

from rlgammon.models.model_errors.model_errors import EligibilityTracesNotInitializedError
from rlgammon.models.value_model import N_INPUT_FEATURES, TDGammonNet

# A constant input vector used to drive a deterministic forward pass.
INPUT_FEATURES = [0.2] * N_INPUT_FEATURES
# The expected TD error for a single step from value p to a bootstrap target one unit higher.
EXPECTED_UNIT_TD_ERROR = 1.0
# Numerical tolerance for the returned TD error.
TD_ERROR_TOLERANCE = 1e-6


def test_init_eligibility_traces_are_zeros_per_parameter() -> None:
    """Test that initialized eligibility traces are zero tensors matching every parameter shape."""
    net = TDGammonNet()
    net.init_eligibility_traces()
    assert net.eligibility_traces is not None
    parameters = list(net.parameters())
    assert len(net.eligibility_traces) == len(parameters)
    for trace, param in zip(net.eligibility_traces, parameters, strict=True):
        assert trace.shape == param.shape
        assert float(trace.abs().sum()) == 0.0


def test_update_weights_without_init_raises() -> None:
    """Test that updating weights before initializing eligibility traces raises."""
    net = TDGammonNet()
    p = net(INPUT_FEATURES)
    try:
        net.update_weights(p, 1.0)
    except EligibilityTracesNotInitializedError:
        return
    msg = "update_weights should raise when eligibility traces are not initialized"
    raise AssertionError(msg)


def test_update_weights_returns_float_and_moves_weights() -> None:
    """Test that a positive-error update returns a python float and moves a weight toward the target."""
    net = TDGammonNet()
    net.init_eligibility_traces()

    p = net(INPUT_FEATURES)
    target = float(p) + 1.0
    before = [param.detach().clone() for param in net.parameters()]

    td_error = net.update_weights(p, target)

    assert isinstance(td_error, float)
    after = list(net.parameters())
    assert any(not th.allclose(b, a) for b, a in zip(before, after, strict=True))


def test_update_weights_is_undiscounted_single_step() -> None:
    """Test that a single step from p=0 to p_next=1 yields a TD error of ~1.0 (no double-decay)."""
    net = TDGammonNet()
    net.init_eligibility_traces()

    # Zero the output bias so the equity head evaluates to exactly 0 (single-win prob 0.5 -> equity 0).
    output_layer = cast("nn.Linear", net.linears[-1])
    with th.no_grad():
        output_layer.bias.zero_()
        output_layer.weight.zero_()
    p = net(INPUT_FEATURES)
    assert abs(float(p)) < TD_ERROR_TOLERANCE

    td_error = net.update_weights(p, 1.0)
    assert abs(td_error - EXPECTED_UNIT_TD_ERROR) < TD_ERROR_TOLERANCE
