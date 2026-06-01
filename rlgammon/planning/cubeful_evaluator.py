"""A cube-/match-aware leaf evaluator that drops into the StarMinimax and StochasticMCTS planners.

OpenSpiel backgammon has no doubling cube, so this evaluator is a pure analytic layer: it reads the
cubeless probability 5-vector from an EQUITY_SIGMOID value network (from the requested perspective)
and converts it, via the Janowski / gnubg model in :mod:`rlgammon.cube.cube_equity`, to a
**per-point** cubeful money equity (money mode) or to ``2 * MWC - 1`` (match mode). Returning the
per-point money equity keeps every leaf value in ``[-3, 3]``, so the star1/star2 bounds of
:class:`~rlgammon.planning.expectiminimax.StarMinimax` remain valid. The cube and match context are
stored from a reference player's view (WHITE by default) and flipped whenever the planner asks for
the opponent's equity (the negamax convention), so the same evaluator serves both sides of the tree.
"""

from rlgammon.cube.cube_equity import cubeful_money_equity, mwc_from_probs
from rlgammon.cube.cube_types import CubeState, GameMode, MatchContext
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.feature_extractor import board_features
from rlgammon.models.model_errors.model_errors import ValueHeadConfigError
from rlgammon.models.model_types import ValueHead
from rlgammon.models.value_model import TDGammonNet
from rlgammon.rlgammon_types import WHITE

# Default gnubg cube-life index for a contact position, mirrored from the equity module.
DEFAULT_CUBE_EFFICIENCY = 0.68


class CubefulEvaluator:
    """Leaf evaluator returning per-point cubeful money equity (or ``2*MWC-1``) from a value network."""

    def __init__(self, model: TDGammonNet, cube_state: CubeState, match_ctx: MatchContext,
                 met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY, *,
                 owner: int = WHITE) -> None:
        """
        Construct the cube-aware evaluator around an EQUITY_SIGMOID value network.

        :param model: the value network (must use the EQUITY_SIGMOID head)
        :param cube_state: the cube state from the reference player's (``owner``'s) perspective
        :param match_ctx: the match context from the reference player's perspective
        :param met: the match-equity table to use (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index passed to the money equity model
        :param owner: the reference player the cube/match are stated for (WHITE=0, BLACK=1)
        :raises ValueHeadConfigError: if the model does not use the EQUITY_SIGMOID head
        """
        if model.value_head != ValueHead.EQUITY_SIGMOID:
            raise ValueHeadConfigError
        self._model = model
        self._cube_state = cube_state
        self._match_ctx = match_ctx
        self._met = met if met is not None else WOOLSEY_HEINRICH
        self._x = x
        self._owner = owner

    def _probs(self, state: GameState, perspective: int) -> list[float]:
        """
        Return a valid cumulative probability 5-vector from ``perspective``'s view of ``state``.

        The raw equity head is grounded only through its scalar combination, so a non-monotone raw
        vector is replaced by the gammonless vector ``[p, 0, 0, 0, 0]`` with ``p`` derived from the
        well-calibrated combined equity, keeping the cube equities meaningful.

        :param state: the game state to evaluate
        :param perspective: the player whose probabilities to compute (WHITE=0, BLACK=1)
        :return: a valid cumulative 5-vector ``(o0, o1, o2, o3, o4)``
        """
        raw = self._model.raw_outputs(board_features(state, perspective))
        o0, o1, o2, o3, o4 = (float(component) for component in raw)
        win_ordered = 1.0 >= o0 >= o1 >= o2 >= 0.0
        lose_ordered = o0 >= o3 >= o4 >= 0.0
        if win_ordered and lose_ordered:
            return [o0, o1, o2, o3, o4]
        equity = (2.0 * o0 - 1.0) + o1 + o2 - o3 - o4
        win_probability = min(max((equity + 1.0) / 2.0, 0.0), 1.0)
        return [win_probability, 0.0, 0.0, 0.0, 0.0]

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s per-point cubeful equity (money) or ``2*MWC-1`` (match) for the state.

        The cube state and match context are flipped to ``perspective``'s view when it is the
        opponent of the reference player, so the value is always expressed from ``perspective``.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the per-point cubeful money equity in ``[-3, 3]``, or ``2*MWC-1`` in match mode
        """
        probs = self._probs(state, perspective)
        cube_state = self._cube_state if perspective == self._owner else self._cube_state.flip_perspective()
        match_ctx = self._match_ctx if perspective == self._owner else self._match_ctx.flip_perspective()
        if match_ctx.mode == GameMode.MATCH:
            mwc = mwc_from_probs(probs, match_ctx, self._met, cube_state, self._x)
            return 2.0 * mwc - 1.0
        # Per-point money equity keeps the value in [-3, 3] regardless of the cube value.
        return cubeful_money_equity(probs, cube_state, self._x) / float(cube_state.value)
