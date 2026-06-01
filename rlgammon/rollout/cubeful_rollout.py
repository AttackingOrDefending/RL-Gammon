"""A cubeful, rollout-backed evaluator: a truncated rollout feeding the analytic doubling-cube layer.

OpenSpiel backgammon has no doubling cube, so -- exactly like :mod:`rlgammon.planning.cubeful_evaluator`
-- this is a pure analytic layer on top of the cubeless rollout and the Janowski / gnubg cube model in
:mod:`rlgammon.cube.cube_equity`. It runs the truncated, variance-reduced
:func:`~rlgammon.rollout.rollout.rollout_equity` to obtain a *sharpened* cubeless equity, reduces that
scalar equity to the gnubg-style gammonless probability vector ``[p, 0, 0, 0, 0]`` with
``p = (equity + 1) / 2`` clamped to ``[0, 1]`` (the documented scalar-to-probability reduction also used
by :class:`~rlgammon.planning.cubeful_evaluator.CubefulEvaluator`, but driven here by the rollout
equity), and returns the **per-point** cubeful money equity (money mode) or ``2 * MWC - 1`` (match
mode). Returning a per-point value keeps every leaf in ``[-3, 3]`` so it remains a drop-in
:class:`~rlgammon.planning.planning_types.Evaluator` for the existing search. The doubling-cube layer
ignores the (zeroed) gammon masses, so a rollout that also estimates gammon rates is left as a future
extension; the win-probability sharpening already drives the cube/take decision through ``p``.
"""

import numpy as np

from rlgammon.cube.cube_equity import cubeful_money_equity, mwc_from_probs
from rlgammon.cube.cube_types import CubeState, GameMode, MatchContext
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.models.model_errors.model_errors import ValueHeadConfigError
from rlgammon.models.model_types import ValueHead
from rlgammon.models.value_model import TDGammonNet
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator
from rlgammon.rlgammon_types import WHITE
from rlgammon.rollout.rollout import RolloutEvaluator
from rlgammon.rollout.rollout_types import RolloutConfig, RolloutPolicy

# Default gnubg cube-life index for a contact position, mirrored from the equity module.
DEFAULT_CUBE_EFFICIENCY = 0.68
# Half-width used to map a cubeless equity in [-1, 1] onto a win probability in [0, 1].
EQUITY_TO_PROB_SCALE = 2.0


class CubefulRolloutEvaluator:
    """Leaf evaluator returning per-point cubeful money equity (or ``2*MWC-1``) from a truncated rollout.

    The underlying cubeless equity is estimated by a truncated, variance-reduced rollout rather than a
    single static forward pass, so the cube and take decisions are driven by the (far more accurate)
    rollout equity. The cube state and match context are stored from a reference player's view (WHITE
    by default) and flipped whenever the opponent's equity is requested, so the same evaluator serves
    both sides of a search tree.
    """

    def __init__(self, model: TDGammonNet, config: RolloutConfig, cube_state: CubeState,
                 match_ctx: MatchContext, met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY, *,
                 owner: int = WHITE, policy: RolloutPolicy | None = None) -> None:
        """
        Construct the cubeful rollout evaluator around an EQUITY_SIGMOID value network.

        :param model: the value network bootstrapped at the truncation leaf (must use EQUITY_SIGMOID)
        :param config: the rollout configuration (trials, truncation depth, variance reduction, ...)
        :param cube_state: the cube state from the reference player's (``owner``'s) perspective
        :param match_ctx: the match context from the reference player's perspective
        :param met: the match-equity table to use (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index passed to the money equity model
        :param owner: the reference player the cube/match are stated for (WHITE=0, BLACK=1)
        :param policy: the move policy the rollout follows; defaults to a 1-ply argmax of the net
        :raises ValueHeadConfigError: if the model does not use the EQUITY_SIGMOID head
        """
        if model.value_head != ValueHead.EQUITY_SIGMOID:
            raise ValueHeadConfigError
        self._model = model
        self._rollout = RolloutEvaluator(ValueNetEvaluator(model), config, policy=policy)
        self._cube_state = cube_state
        self._match_ctx = match_ctx
        self._met = met if met is not None else WOOLSEY_HEINRICH
        self._x = x
        self._owner = owner

    def _rollout_probs(self, state: GameState, perspective: int) -> list[float]:
        """
        Return a valid cumulative probability 5-vector for ``state`` from the rollout equity.

        The truncated rollout yields a sharpened cubeless equity ``e`` in ``[-3, 3]``; it is mapped to
        a win probability ``p = (e + 1) / 2`` clamped to ``[0, 1]`` and returned as the gnubg-style
        gammonless vector ``[p, 0, 0, 0, 0]``. This mirrors the documented reduction used by
        :class:`~rlgammon.planning.cubeful_evaluator.CubefulEvaluator`, but driven by the (more
        accurate) rollout equity instead of a single static evaluation.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose probabilities to compute (WHITE=0, BLACK=1)
        :return: a valid cumulative 5-vector ``(o0, o1, o2, o3, o4)``
        """
        equity = self._rollout.evaluate(state, perspective)
        win_probability = min(max((equity + 1.0) / EQUITY_TO_PROB_SCALE, 0.0), 1.0)
        return [win_probability, 0.0, 0.0, 0.0, 0.0]

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s per-point cubeful equity (money) or ``2*MWC-1`` (match) for the state.

        The cube state and match context are flipped to ``perspective``'s view when it is the opponent
        of the reference player, so the value is always expressed from ``perspective``.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the per-point cubeful money equity in ``[-3, 3]``, or ``2*MWC-1`` in match mode
        """
        probs = self._rollout_probs(state, perspective)
        cube_state = self._cube_state if perspective == self._owner else self._cube_state.flip_perspective()
        match_ctx = self._match_ctx if perspective == self._owner else self._match_ctx.flip_perspective()
        if match_ctx.mode == GameMode.MATCH:
            mwc = mwc_from_probs(probs, match_ctx, self._met, cube_state, self._x)
            return EQUITY_TO_PROB_SCALE * mwc - 1.0
        # Per-point money equity keeps the value in [-3, 3] regardless of the cube value.
        return cubeful_money_equity(probs, cube_state, self._x) / float(cube_state.value)

    def cubeful_equity(self, state: GameState, perspective: int) -> float:
        """
        Return the full (cube-scaled) cubeful money equity for ``state`` from ``perspective``.

        Unlike :meth:`evaluate` (which divides out the cube value to stay in ``[-3, 3]`` for search),
        this returns the equity at the actual stake, suitable for reporting a money game's swing.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose cubeful equity to return (WHITE=0, BLACK=1)
        :return: the on-stake cubeful money equity in points
        """
        probs = self._rollout_probs(state, perspective)
        cube_state = self._cube_state if perspective == self._owner else self._cube_state.flip_perspective()
        return cubeful_money_equity(probs, cube_state, self._x)


def cubeful_rollout_money_equity(model: TDGammonNet, state: GameState, config: RolloutConfig,
                                 cube_state: CubeState, *, perspective: int = WHITE,
                                 x: float = DEFAULT_CUBE_EFFICIENCY,
                                 rng: np.random.Generator | None = None) -> float:
    """
    Return the cube-scaled cubeful money equity for a position from a truncated rollout (convenience).

    :param model: the EQUITY_SIGMOID value network bootstrapped at the truncation leaf
    :param state: the (decision-node) game state to evaluate
    :param config: the rollout configuration
    :param cube_state: the cube state from ``perspective``'s view
    :param perspective: the player whose cubeful equity to return (WHITE=0, BLACK=1)
    :param x: the cube-life index
    :param rng: an optional generator seeding the rollout (defaults to ``config.seed``)
    :return: the on-stake cubeful money equity in points from ``perspective``
    :raises ValueHeadConfigError: if the model does not use the EQUITY_SIGMOID head
    """
    if model.value_head != ValueHead.EQUITY_SIGMOID:
        raise ValueHeadConfigError
    del rng  # the evaluator manages its own reproducible generator from the config seed
    evaluator = CubefulRolloutEvaluator(model, config, cube_state, MatchContext(mode=GameMode.MONEY),
                                        x=x, owner=perspective)
    return evaluator.cubeful_equity(state, perspective)
