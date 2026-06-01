"""Truncated rollouts with variance reduction over the cubeless game and value network.

A static value net evaluates a position in one shot; a *truncated rollout* evaluates it far more
precisely by playing many short, independent playouts from it (following a move policy, resolving the
true dice distribution) and bootstrapping the value net at the truncation leaf. This is the core
accuracy technique GNU Backgammon and ExtremeGammon use. The package adds a control-variate
("lookahead") variance reduction -- the rollout estimates the *correction* to the analytically
averaged static look-ahead rather than the raw outcome -- and an optional antithetic-dice scheme, and
exposes a :class:`~rlgammon.planning.planning_types.Evaluator` so a rollout-backed evaluation drops
into the existing search and agents as a stronger (slower) test-time evaluator.

The package mirrors the structure of the other algorithm packages: a ``rollout_types`` module of the
frozen ``RolloutConfig`` / ``RolloutResult`` value objects and the ``RolloutPolicy`` protocol, a
``rollout_errors`` subpackage and the ``rollout`` module with the estimator and the evaluator.
"""

from rlgammon.rollout.cubeful_rollout import CubefulRolloutEvaluator, cubeful_rollout_money_equity
from rlgammon.rollout.rollout import RolloutEvaluator, rollout_equity
from rlgammon.rollout.rollout_types import (
    RolloutConfig,
    RolloutPolicy,
    RolloutResult,
)

__all__ = [
    "CubefulRolloutEvaluator",
    "RolloutConfig",
    "RolloutEvaluator",
    "RolloutPolicy",
    "RolloutResult",
    "cubeful_rollout_money_equity",
    "rollout_equity",
]
