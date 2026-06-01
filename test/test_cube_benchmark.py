"""Smoke and structure tests for the model-agnostic cube-quality / calibration benchmark.

The cube-context table is exercised deterministically on fixed probability vectors (no value
network, no OpenSpiel needed), proving the four contexts are present and that the Crawford context
is dead. The full self-play benchmark is gated on OpenSpiel: when ``pyspiel`` is available a tiny
fresh-network run asserts the report carries calibration samples, reliability bins and the cube
tables; without it the benchmark falls back to a fixed-probability cube table whose structure is
still checked.
"""

import pytest

from rlgammon.cube.cube_equity import CubeAction, TakeAction
from rlgammon.cube.cube_types import GameMode
from rlgammon.game import is_openspiel_available
from scripts.cube_benchmark import (
    FALLBACK_CUBE_PROBS,
    BenchmarkReport,
    CubeContext,
    cube_contexts,
    cube_row,
    format_report,
    run_benchmark,
)

# A tiny number of self-play games for the fast OpenSpiel smoke test.
SMOKE_GAMES = 4
# The number of named contexts compared in the headline cube-decision table.
EXPECTED_CONTEXTS = 4
# A strong gammonless position that is a clear money double/take (used for the take assertion).
STRONG_TAKE_PROBS = [0.70, 0.0, 0.0, 0.0, 0.0]
# The label of the Crawford context (the cube must be dead there).
CRAWFORD_CONTEXT_NAME = "match 0-6 Crawford"
# The label of the post-Crawford context (the cube is live again there).
POST_CRAWFORD_CONTEXT_NAME = "match 0-6 post-Crawford"


def _context_by_name(name: str) -> CubeContext:
    """Return the named cube context from the benchmark's context list.

    :param name: the context label to look up
    :return: the matching cube context
    """
    return next(context for context in cube_contexts() if context.name == name)


def test_four_contexts_present() -> None:
    """Test that the headline cube table compares exactly the four documented contexts."""
    contexts = cube_contexts()
    assert len(contexts) == EXPECTED_CONTEXTS
    names = [context.name for context in contexts]
    assert CRAWFORD_CONTEXT_NAME in names
    assert POST_CRAWFORD_CONTEXT_NAME in names
    # Exactly one context is money play; the rest are match play.
    money = [context for context in contexts if context.match_ctx.mode == GameMode.MONEY]
    assert len(money) == 1


def test_crawford_context_yields_no_double() -> None:
    """Test that the Crawford context produces NO_DOUBLE regardless of how strong the position is."""
    crawford = _context_by_name(CRAWFORD_CONTEXT_NAME)
    assert crawford.match_ctx.cube_dead_this_game is True
    for probs in ([0.5, 0.0, 0.0, 0.0, 0.0], STRONG_TAKE_PROBS, [0.9, 0.8, 0.1, 0.0, 0.0]):
        row = cube_row(probs, crawford)
        assert row.action == CubeAction.NO_DOUBLE
        assert row.value_label == "mwc"


def test_post_crawford_trailer_doubles() -> None:
    """Test that the post-Crawford trailer doubles a strong position (the cube is live again)."""
    post_crawford = _context_by_name(POST_CRAWFORD_CONTEXT_NAME)
    assert post_crawford.match_ctx.cube_dead_this_game is False
    row = cube_row(STRONG_TAKE_PROBS, post_crawford)
    assert row.action in (CubeAction.DOUBLE_TAKE, CubeAction.DOUBLE_PASS)


def test_money_context_reports_equity_and_take() -> None:
    """Test that the money context reports a money equity and a definite take decision."""
    money = _context_by_name("money")
    row = cube_row(STRONG_TAKE_PROBS, money)
    assert row.value_label == "equity"
    assert row.take in (TakeAction.TAKE, TakeAction.PASS)


def test_run_benchmark_structure_without_openspiel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the benchmark falls back to a fixed-probs cube table when OpenSpiel is unavailable.

    OpenSpiel availability is forced off so this path is covered even where ``pyspiel`` is installed;
    the fallback report must skip calibration but still carry a complete four-context cube table.
    """
    monkeypatch.setattr("scripts.cube_benchmark.is_openspiel_available", lambda: False)
    report = run_benchmark(model_path=None, fresh=True, games=SMOKE_GAMES, seed=0)
    assert isinstance(report, BenchmarkReport)
    assert report.openspiel is False
    assert report.calibration is None
    assert len(report.positions) == 1
    position = report.positions[0]
    assert position.raw_probs == FALLBACK_CUBE_PROBS
    assert len(position.rows) == EXPECTED_CONTEXTS
    crawford_row = next(row for row in position.rows if row.context_name == CRAWFORD_CONTEXT_NAME)
    assert crawford_row.action == CubeAction.NO_DOUBLE
    # The formatted report must mention every report section.
    text = format_report(report)
    assert "1. Probability calibration" in text
    assert "2. Cube decisions are match-score-dependent" in text
    assert "3. Probability path used by the cube layer" in text


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is not installed")
def test_run_benchmark_smoke_with_openspiel() -> None:
    """Test that the full self-play benchmark runs end-to-end on a fresh net and is well-formed.

    A fresh (untrained, scalar-like) network is expected to be uncalibrated, so the reliability table
    is populated but the assertions only check structure, not calibration quality.
    """
    report = run_benchmark(model_path=None, fresh=True, games=SMOKE_GAMES, seed=0)
    assert report.openspiel is True
    assert report.calibration is not None
    assert report.calibration.n_samples > 0
    assert len(report.calibration.mean_predicted) == len(FALLBACK_CUBE_PROBS)
    assert report.calibration.reliability  # at least one non-empty reliability bin
    # The opening position plus the sampled positions are all reported, each with four contexts.
    assert len(report.positions) >= 1
    for position in report.positions:
        assert len(position.rows) == EXPECTED_CONTEXTS
        crawford_row = next(row for row in position.rows if row.context_name == CRAWFORD_CONTEXT_NAME)
        assert crawford_row.action == CubeAction.NO_DOUBLE
    # A fresh scalar-like net has non-monotone raw components, so the cube layer falls back.
    assert any(not position.used_raw for position in report.positions)
    # The formatted report is non-empty and contains the calibration header.
    assert "Brier score" in format_report(report)
