"""Frozen value types and the policy protocol for the truncated-rollout package.

The rollout package is a pure analytic layer on top of the game engine, the value-network
:class:`~rlgammon.planning.planning_types.Evaluator` and a move ``RolloutPolicy``; nothing in this
module touches torch or ``pyspiel``. ``RolloutConfig`` bundles the (immutable) rollout
hyper-parameters and ``RolloutResult`` is the immutable estimate a rollout returns.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rlgammon.game.backgammon_protocol import GameState

# Default number of independent truncated playouts per rollout.
DEFAULT_NUM_TRIALS = 256
# Default truncation length: the number of decision plies played before bootstrapping with the net.
DEFAULT_MAX_DEPTH = 7
# Default seed for the rollout random number generator.
DEFAULT_SEED = 0
# Sentinel ``control_variate_depth``: read the control-variate afterstate one ply before truncation
# (``max_depth - 1``, clamped to >= 1). The net's prediction nearest the truncation leaf correlates
# most strongly with the bootstrapped outcome, so this auto-depth maximises the variance reduction.
AUTO_CONTROL_VARIATE_DEPTH = 0


@runtime_checkable
class RolloutPolicy(Protocol):
    """Protocol for the move policy a rollout follows at every decision node it visits."""

    def choose_move(self, actions: list[int], state: GameState) -> int:
        """
        Return the action the policy plays at ``state`` from among ``actions``.

        :param actions: the legal action ids at the (decision-node) ``state``
        :param state: the current decision-node game state
        :return: the chosen action id
        """


@dataclass(frozen=True)
class RolloutConfig:
    """
    Immutable bundle of truncated-rollout hyper-parameters.

    :param num_trials: the number of independent truncated playouts to average
    :param max_depth: the truncation length in decision plies before bootstrapping with the evaluator
    :param seed: the seed for the rollout random number generator (reproducibility / common randoms)
    :param variance_reduction: whether to apply the control-variate (lookahead) variance reduction
    :param antithetic: whether to additionally pair each trial with an antithetic-dice partner trial
        (a measure-preserving inverse-CDF reflection; exact for the uniform backgammon dice)
    :param control_variate_depth: the decision-ply depth of the afterstate read as the control-variate
        baseline; ``AUTO_CONTROL_VARIATE_DEPTH`` (0) reads it one ply before truncation (the strongest
        baseline), otherwise must satisfy ``1 <= control_variate_depth <= max_depth``
    :param cubeful: whether the result should be interpreted as a cubeful equity (see the evaluator)
    """

    num_trials: int = DEFAULT_NUM_TRIALS
    max_depth: int = DEFAULT_MAX_DEPTH
    seed: int = DEFAULT_SEED
    variance_reduction: bool = True
    antithetic: bool = False
    control_variate_depth: int = AUTO_CONTROL_VARIATE_DEPTH
    cubeful: bool = False

    def resolved_control_variate_depth(self) -> int:
        """
        Return the effective control-variate depth, resolving the ``AUTO`` sentinel.

        :return: ``max(max_depth - 1, 1)`` for the auto sentinel, otherwise ``control_variate_depth``
        """
        if self.control_variate_depth == AUTO_CONTROL_VARIATE_DEPTH:
            return max(self.max_depth - 1, 1)
        return self.control_variate_depth


@dataclass(frozen=True)
class RolloutResult:
    """
    Immutable summary of a truncated rollout from a single root decision node.

    :param equity: the estimated equity (in points) from the requested perspective
    :param std_error: the standard error of ``equity`` (the standard deviation of the trial estimator
        divided by ``sqrt(num_trials)``)
    :param num_trials: the number of trials (or antithetic pairs) the estimate is averaged over
    :param baseline: the static evaluator equity at the root (the control-variate baseline)
    :param variance_reduced: whether the control-variate variance reduction was applied
    """

    equity: float
    std_error: float
    num_trials: int
    baseline: float
    variance_reduced: bool
