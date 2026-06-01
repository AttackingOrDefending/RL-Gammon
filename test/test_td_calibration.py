"""Tests for the calibrated multi-output TD(lambda) update of the equity head.

These cover the outcome-vector encoding (:meth:`TDGammonNet.outcome_target`), the per-output
eligibility traces (:meth:`TDGammonNet.init_outcome_traces`) and, most importantly, that the
multi-output update (:meth:`TDGammonNet.update_outcome_weights`) actually calibrates the individual
probability components toward a fixed target -- the property the scalar update cannot provide.
"""
import torch as th

from rlgammon.models.value_model import (
    N_EQUITY_COMPONENTS,
    N_INPUT_FEATURES,
    TDGammonNet,
)

# The exact cumulative outcome vectors expected for every WHITE-centric terminal return.
EXPECTED_OUTCOME_VECTORS = {
    1: [1.0, 0.0, 0.0, 0.0, 0.0],
    2: [1.0, 1.0, 0.0, 0.0, 0.0],
    3: [1.0, 1.0, 1.0, 0.0, 0.0],
    -1: [0.0, 0.0, 0.0, 0.0, 0.0],
    -2: [0.0, 0.0, 0.0, 1.0, 0.0],
    -3: [0.0, 0.0, 0.0, 1.0, 1.0],
}
# Numerical tolerance for the combine_equity round-trip.
EQUITY_TOLERANCE = 1e-6
# Hidden width of the tiny network used in the calibration test.
TEST_HIDDEN = 16
# Number of calibration iterations driving the components toward the fixed target.
CALIBRATION_ITERS = 300
# A modest learning rate for the calibration test.
CALIBRATION_LR = 0.05
# The fixed gammon-win target the calibration test drives toward.
GAMMON_WIN_TARGET = [1.0, 1.0, 0.0, 0.0, 0.0]
# A predicted win-component value clearly calibrated toward 1.
CALIBRATED_HIGH = 0.8
# A predicted loss-component value that must stay clearly low (near 0).
CALIBRATED_LOW = 0.2


def test_outcome_target_returns_exact_vectors_and_round_trips() -> None:
    """Test the exact 6 outcome vectors and that combine_equity(outcome_target(r)) == r."""
    for returns_white, expected in EXPECTED_OUTCOME_VECTORS.items():
        target = TDGammonNet.outcome_target(float(returns_white))
        assert target.shape == (N_EQUITY_COMPONENTS,)
        assert target.tolist() == expected
        assert abs(float(TDGammonNet.combine_equity(target)) - returns_white) < EQUITY_TOLERANCE


def test_init_outcome_traces_are_zeros_per_output_per_parameter() -> None:
    """Test that outcome traces are zero tensors of shape (5, *param.shape) for every parameter."""
    net = TDGammonNet(hidden=TEST_HIDDEN)
    net.init_outcome_traces()
    assert net.outcome_traces is not None
    parameters = list(net.parameters())
    assert len(net.outcome_traces) == len(parameters)
    for trace, param in zip(net.outcome_traces, parameters, strict=True):
        assert trace.shape == (N_EQUITY_COMPONENTS, *param.shape)
        assert float(trace.abs().sum()) == 0.0


def test_update_outcome_weights_calibrates_components_toward_target() -> None:
    """Test that the multi-output update moves each component toward a fixed gammon-win target."""
    th.manual_seed(0)
    net = TDGammonNet(lr=CALIBRATION_LR, hidden=TEST_HIDDEN, seed=0)
    net.init_outcome_traces()

    feature = th.rand(N_INPUT_FEATURES).tolist()
    target = th.tensor(GAMMON_WIN_TARGET)
    before = net.raw_outputs(feature).detach().clone()

    first_loss = net.update_outcome_weights(net.raw_outputs(feature), target.detach())
    last_loss = first_loss
    for _ in range(CALIBRATION_ITERS - 1):
        last_loss = net.update_outcome_weights(net.raw_outputs(feature), target.detach())

    after = net.raw_outputs(feature).detach()

    # The loss is a python float and the components have clearly moved toward the gammon-win target.
    assert isinstance(first_loss, float)
    assert last_loss < first_loss
    assert after[0] > before[0]
    assert after[1] > before[1]
    assert after[0] > CALIBRATED_HIGH
    assert after[1] > CALIBRATED_HIGH
    assert after[3] < CALIBRATED_LOW
    assert after[4] < CALIBRATED_LOW
