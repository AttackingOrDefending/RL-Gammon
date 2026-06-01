"""Pure-float Janowski / gnubg doubling-cube equities and cube decisions (no torch, no game engine).

Every function here consumes a cubeless probability 5-vector ``(o0, o1, o2, o3, o4)`` of cumulative
sigmoids (as produced by ``TDGammonNet.raw_outputs`` for the EQUITY_SIGMOID head):

* ``o0`` = P(win any),               ``o1`` = P(win >= gammon),  ``o2`` = P(win backgammon),
* ``o3`` = P(lose >= gammon),        ``o4`` = P(lose backgammon).

The money path implements the exact Janowski cube-life model; the match path converts the discrete
outcome masses to match-winning chance through a match-equity table under a dead-cube approximation
(the cube is held at its current value when scoring the outcome), which is documented at the call site.
"""

from enum import Enum, auto

from rlgammon.cube.cube_errors.cube_errors import InvalidProbabilityVectorError
from rlgammon.cube.cube_types import INITIAL_CUBE_VALUE, CubeOwner, CubeState, GameMode, MatchContext
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH

# Default gnubg cube-life / cube-efficiency index for a contact position.
DEFAULT_CUBE_EFFICIENCY = 0.68
# Cube-efficiency endpoints for the optional race interpolation (pips -> efficiency).
RACE_EFFICIENCY_LOW = 0.6
RACE_EFFICIENCY_HIGH = 0.7
RACE_PIPS_LOW = 40.0
RACE_PIPS_HIGH = 120.0
# Probabilities within this tolerance of a certain win/loss are treated as decided (avoid 0/0).
PROBABILITY_EPSILON = 1e-9
# Points won/lost for a single game, a gammon and a backgammon.
SINGLE_POINTS = 1.0
GAMMON_POINTS = 2.0
BACKGAMMON_POINTS = 3.0
# The denominator constant in Janowski's centred-cube equity (4 / (4 - x)).
CENTERED_DENOMINATOR_BASE = 4.0
# Number of components in a cubeless probability vector.
N_PROBABILITY_COMPONENTS = 5


class CubeAction(Enum):
    """The doubler's cube decision for a position."""

    NO_DOUBLE = auto()
    DOUBLE_TAKE = auto()
    DOUBLE_PASS = auto()
    TOO_GOOD = auto()


class TakeAction(Enum):
    """The taker's response to an offered double."""

    TAKE = auto()
    PASS = auto()


def _validate(probs: list[float]) -> None:
    """
    Validate that a probability vector has the expected length.

    :param probs: the cubeless probability 5-vector to validate
    :raises InvalidProbabilityVectorError: if the vector does not hold exactly 5 components
    """
    if len(probs) != N_PROBABILITY_COMPONENTS:
        raise InvalidProbabilityVectorError


def cubeless_equity(probs: list[float]) -> float:
    """
    Return the cubeless money equity ``(2*o0-1) + o1 + o2 - o3 - o4`` for a probability vector.

    This is identical to ``TDGammonNet.combine_equity`` and lies in ``[-3, 3]``.

    :param probs: the cubeless probability 5-vector ``(o0, o1, o2, o3, o4)``
    :return: the cubeless money equity in points
    """
    _validate(probs)
    o0, o1, o2, o3, o4 = probs
    return (2.0 * o0 - 1.0) + o1 + o2 - o3 - o4


def w_and_l(probs: list[float]) -> tuple[float, float]:
    """
    Return ``(W, L)``: the average points won given a win and lost given a loss.

    The cumulative vector is decomposed into mutually-exclusive masses and averaged within the
    winning and losing branches. When a win (or loss) is essentially impossible the corresponding
    average defaults to a single point so the downstream formulas stay finite.

    :param probs: the cubeless probability 5-vector ``(o0, o1, o2, o3, o4)``
    :return: a tuple ``(W, L)`` of the average magnitude of a win and of a loss (both >= 1)
    """
    _validate(probs)
    o0, o1, o2, o3, o4 = probs
    p_win = o0
    p_lose = 1.0 - o0
    p_win_single = o0 - o1
    p_win_gammon = o1 - o2
    p_win_bg = o2
    p_lose_single = p_lose - o3
    p_lose_gammon = o3 - o4
    p_lose_bg = o4
    if p_win <= PROBABILITY_EPSILON:
        avg_win = SINGLE_POINTS
    else:
        avg_win = (SINGLE_POINTS * p_win_single + GAMMON_POINTS * p_win_gammon
                   + BACKGAMMON_POINTS * p_win_bg) / p_win
    if p_lose <= PROBABILITY_EPSILON:
        avg_lose = SINGLE_POINTS
    else:
        avg_lose = (SINGLE_POINTS * p_lose_single + GAMMON_POINTS * p_lose_gammon
                    + BACKGAMMON_POINTS * p_lose_bg) / p_lose
    return avg_win, avg_lose


def cube_efficiency(pips: float | None = None) -> float:
    """
    Return the gnubg cube-life index ``x``, optionally interpolated by an estimated race length.

    With no pip estimate the default contact efficiency ``0.68`` is returned. With a pip estimate
    the efficiency is linearly interpolated from ``0.6`` at 40 pips to ``0.7`` at 120 pips and
    clamped to ``[0.6, 0.7]`` (longer races make the cube more valuable / efficient).

    :param pips: an optional estimate of the on-roll player's pip count, or ``None`` if unknown
    :return: the cube-life index ``x`` in ``[0, 1]``
    """
    if pips is None:
        return DEFAULT_CUBE_EFFICIENCY
    span = RACE_PIPS_HIGH - RACE_PIPS_LOW
    fraction = (pips - RACE_PIPS_LOW) / span
    fraction = min(max(fraction, 0.0), 1.0)
    return RACE_EFFICIENCY_LOW + fraction * (RACE_EFFICIENCY_HIGH - RACE_EFFICIENCY_LOW)


def take_point(avg_win: float, avg_lose: float, x: float = DEFAULT_CUBE_EFFICIENCY) -> float:
    """
    Return the take point ``(L - 0.5) / (W + L + 0.5x)``: the minimum win probability to take.

    :param avg_win: the average magnitude of a win (``W``)
    :param avg_lose: the average magnitude of a loss (``L``)
    :param x: the cube-life index
    :return: the take-point win probability
    """
    return (avg_lose - 0.5) / (avg_win + avg_lose + 0.5 * x)


def cash_point(avg_win: float, avg_lose: float, x: float = DEFAULT_CUBE_EFFICIENCY) -> float:
    """
    Return the cash point ``(L + 0.5 + 0.5x) / (W + L + 0.5x)``: the win probability to cash a double.

    :param avg_win: the average magnitude of a win (``W``)
    :param avg_lose: the average magnitude of a loss (``L``)
    :param x: the cube-life index
    :return: the cash-point win probability
    """
    return (avg_lose + 0.5 + 0.5 * x) / (avg_win + avg_lose + 0.5 * x)


def _is_live_jacoby_centered_one(cube_state: CubeState) -> bool:
    """
    Return whether the Jacoby rule has collapsed gammons (centred 1-cube under live Jacoby).

    Under the Jacoby rule an undoubled game can only be won/lost a single point, so the
    gammon/backgammon masses must be ignored when valuing a centred 1-cube money game.

    :param cube_state: the cube state to test
    :return: whether the centred-1 gammonless special case applies
    """
    return (cube_state.jacoby and cube_state.owner == CubeOwner.CENTERED
            and cube_state.value <= INITIAL_CUBE_VALUE)


def cubeful_money_equity(probs: list[float], cube_state: CubeState,
                         x: float = DEFAULT_CUBE_EFFICIENCY) -> float:
    """
    Return the Janowski cubeful money equity (in points) from the on-roll player's perspective.

    The equity is scaled by the cube value and dispatched on cube ownership:

    * owner ``ME``       : ``C * [ p*(W+L+0.5x) - L ]``
    * owner ``OPP``      : ``C * [ p*(W+L+0.5x) - L - 0.5x ]``
    * owner ``CENTERED`` : ``(4C/(4-x)) * [ p*(W+L+0.5x) - L - 0.25x ]``

    A centred 1-cube under the live Jacoby rule collapses gammons (``W = L = 1``), giving the
    cubeless money line ``C * (2p - 1)``.

    :param probs: the cubeless probability 5-vector ``(o0, o1, o2, o3, o4)``
    :param cube_state: the cube value, owner and rules from the on-roll player's perspective
    :param x: the cube-life index
    :return: the on-roll player's cubeful money equity in points
    """
    _validate(probs)
    p = probs[0]
    cube_value = float(cube_state.value)
    if _is_live_jacoby_centered_one(cube_state):
        # Live Jacoby on a centred 1-cube: only single games count, so W = L = 1 and E = C*(2p - 1).
        return cube_value * (2.0 * p - 1.0)
    avg_win, avg_lose = w_and_l(probs)
    base = p * (avg_win + avg_lose + 0.5 * x) - avg_lose
    if cube_state.owner == CubeOwner.ME:
        return cube_value * base
    if cube_state.owner == CubeOwner.OPP:
        return cube_value * (base - 0.5 * x)
    centered_factor = CENTERED_DENOMINATOR_BASE / (CENTERED_DENOMINATOR_BASE - x)
    return cube_value * centered_factor * (base - 0.25 * x)


def _score_outcome_mwc(match_ctx: MatchContext, met: MET, my_points: float, opp_points: float) -> float:
    """
    Return the on-roll player's MWC after a game scored ``my_points`` to me / ``opp_points`` to the opponent.

    Exactly one of the two point totals is non-zero (the winner's stake). Scores are capped at the
    match length and the resulting away-counts are looked up in the match-equity table.

    :param match_ctx: the pre-game match context (the on-roll player's scores)
    :param met: the match-equity table to read MWC from
    :param my_points: the points the on-roll player would win (0 if they lose)
    :param opp_points: the points the opponent would win (0 if the on-roll player wins)
    :return: the on-roll player's match-winning chance after the outcome
    """
    new_my_score = min(match_ctx.my_score + round(my_points), match_ctx.match_length)
    new_opp_score = min(match_ctx.opp_score + round(opp_points), match_ctx.match_length)
    if new_my_score >= match_ctx.match_length:
        return 1.0
    if new_opp_score >= match_ctx.match_length:
        return 0.0
    my_away = max(match_ctx.match_length - new_my_score, 1)
    opp_away = max(match_ctx.match_length - new_opp_score, 1)
    return met.mwc_for_away(my_away, opp_away)


def mwc_from_probs(probs: list[float], match_ctx: MatchContext, met: MET = WOOLSEY_HEINRICH,
                   cube_state: CubeState | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> float:
    """
    Return the on-roll player's match-winning chance for a position (dead-cube approximation).

    The discrete outcome masses (win/lose single, gammon, backgammon) are each multiplied by the
    cube value to obtain the points at stake, the resulting match score is looked up in the table,
    and the MWCs are probability-weighted. This is a dead-cube model: the cube is held at its
    current value for the purpose of scoring the outcome (the exact cubeful recursion is not used).
    The ``x`` argument is accepted for interface symmetry with the money path and is unused here.

    :param probs: the cubeless probability 5-vector ``(o0, o1, o2, o3, o4)``
    :param match_ctx: the pre-game match context (the on-roll player's scores)
    :param met: the match-equity table to read MWC from
    :param cube_state: the cube state (only its value scales the stake); ``None`` means a 1-cube
    :param x: accepted for symmetry with the money path; unused in this dead-cube approximation
    :return: the on-roll player's match-winning chance in ``[0, 1]``
    """
    del x
    _validate(probs)
    o0, o1, o2, o3, o4 = probs
    cube_value = float(cube_state.value) if cube_state is not None else 1.0
    p_win_single = o0 - o1
    p_win_gammon = o1 - o2
    p_win_bg = o2
    p_lose_single = (1.0 - o0) - o3
    p_lose_gammon = o3 - o4
    p_lose_bg = o4
    win_masses = ((p_win_single, SINGLE_POINTS), (p_win_gammon, GAMMON_POINTS), (p_win_bg, BACKGAMMON_POINTS))
    lose_masses = ((p_lose_single, SINGLE_POINTS), (p_lose_gammon, GAMMON_POINTS), (p_lose_bg, BACKGAMMON_POINTS))
    mwc = 0.0
    for mass, points in win_masses:
        mwc += mass * _score_outcome_mwc(match_ctx, met, cube_value * points, 0.0)
    for mass, points in lose_masses:
        mwc += mass * _score_outcome_mwc(match_ctx, met, 0.0, cube_value * points)
    return mwc


def _doubler_value(probs: list[float], match_ctx: MatchContext | None, met: MET,
                   cube_state: CubeState, x: float) -> float:
    """
    Return the doubler-side value (cubeful money equity, or MWC in match mode) for a cube state.

    :param probs: the cubeless probability 5-vector from the doubler's perspective
    :param match_ctx: the match context for match mode, or ``None`` for money mode
    :param met: the match-equity table (used only in match mode)
    :param cube_state: the cube state from the doubler's perspective
    :param x: the cube-life index
    :return: the doubler's equity (money points) or match-winning chance
    """
    if match_ctx is not None and match_ctx.mode == GameMode.MATCH:
        return mwc_from_probs(probs, match_ctx, met, cube_state, x)
    return cubeful_money_equity(probs, cube_state, x)


def double_decision(probs: list[float], cube_state: CubeState, match_ctx: MatchContext | None = None,
                    *, met: MET = WOOLSEY_HEINRICH, x: float = DEFAULT_CUBE_EFFICIENCY) -> CubeAction:
    """
    Return the doubler's cube action by comparing the three gnubg cubeful equities.

    The no-double, double-take and double-pass values are computed from the doubler's perspective
    (in money equity, or in match-winning chance for a match). If no-double already dominates the
    pass value the position is :class:`CubeAction.TOO_GOOD` (play on for the gammon); otherwise a
    double is offered iff the better of take/pass exceeds no-double, labelled
    :class:`CubeAction.DOUBLE_PASS` when a pass is at least as good for the doubler as a take, else
    :class:`CubeAction.DOUBLE_TAKE`. In a Crawford game (cube dead) the action is always
    :class:`CubeAction.NO_DOUBLE`.

    :param probs: the cubeless probability 5-vector from the doubler's perspective
    :param cube_state: the current cube state from the doubler's perspective
    :param match_ctx: the match context for match mode, or ``None`` for money mode
    :param met: the match-equity table (used only in match mode)
    :param x: the cube-life index
    :return: the doubler's cube action
    """
    if match_ctx is not None and match_ctx.cube_dead_this_game:
        return CubeAction.NO_DOUBLE
    if not cube_state.can_double():
        return CubeAction.NO_DOUBLE

    e_nd = _doubler_value(probs, match_ctx, met, cube_state, x)
    cube_after = cube_state.after_double()
    e_dt = _doubler_value(probs, match_ctx, met, cube_after, x)
    e_dp = _pass_value(cube_state, match_ctx, met)

    if e_nd >= e_dp:
        return CubeAction.TOO_GOOD
    value_if_double = min(e_dt, e_dp)
    if value_if_double <= e_nd:
        return CubeAction.NO_DOUBLE
    return CubeAction.DOUBLE_PASS if e_dp <= e_dt else CubeAction.DOUBLE_TAKE


def _pass_value(cube_state: CubeState, match_ctx: MatchContext | None, met: MET) -> float:
    """
    Return the doubler's value when the opponent passes the double.

    In money play a pass wins exactly the current cube value. In match play the doubler scores the
    current cube value as points and the resulting MWC is read from the table.

    :param cube_state: the current cube state from the doubler's perspective
    :param match_ctx: the match context for match mode, or ``None`` for money mode
    :param met: the match-equity table (used only in match mode)
    :return: the doubler's value if the opponent passes
    """
    if match_ctx is not None and match_ctx.mode == GameMode.MATCH:
        return _score_outcome_mwc(match_ctx, met, float(cube_state.value), 0.0)
    return float(cube_state.value)


def take_decision(probs_taker: list[float], cube_state: CubeState,
                  match_ctx: MatchContext | None = None, *, met: MET = WOOLSEY_HEINRICH,
                  x: float = DEFAULT_CUBE_EFFICIENCY) -> TakeAction:
    """
    Return the taker's response to an offered double.

    The taker compares taking (owning the doubled cube) against passing (conceding the current cube
    value). All probabilities are from the taker's perspective.

    :param probs_taker: the cubeless probability 5-vector from the taker's perspective
    :param cube_state: the current cube state from the taker's perspective (pre-double)
    :param match_ctx: the match context for match mode, or ``None`` for money mode
    :param met: the match-equity table (used only in match mode)
    :param x: the cube-life index
    :return: the taker's response (:class:`TakeAction.TAKE` or :class:`TakeAction.PASS`)
    """
    # After the double the cube value has doubled and the taker owns it (owner == ME for the taker).
    cube_after_value = cube_state.value * 2
    cube_taker_owned = CubeState(value=cube_after_value, owner=CubeOwner.ME, jacoby=cube_state.jacoby,
                                 beavers=cube_state.beavers, max_cube=cube_state.max_cube)
    if match_ctx is not None and match_ctx.mode == GameMode.MATCH:
        e_take = mwc_from_probs(probs_taker, match_ctx, met, cube_taker_owned, x)
        # A pass concedes the current (pre-double) cube value to the opponent.
        e_pass = _score_outcome_mwc(match_ctx, met, 0.0, float(cube_state.value))
    else:
        e_take = cubeful_money_equity(probs_taker, cube_taker_owned, x)
        e_pass = -float(cube_state.value)
    return TakeAction.TAKE if e_take > e_pass else TakeAction.PASS
