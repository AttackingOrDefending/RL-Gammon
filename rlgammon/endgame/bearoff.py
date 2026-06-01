"""Exact one-sided bear-off database and an exact, gammon-aware race win probability.

This is the analogue of GNU Backgammon's one-sided bear-off database. For a single side whose 15
checkers are all in its home board, :func:`bearoff_distribution` returns the exact probability
distribution over the number of *rolls* needed to bear all 15 off, assuming the side plays its dice
to minimise the expected number of remaining rolls. The distribution is computed by dynamic
programming over the home-board configuration: every one of the 21 distinct dice rolls (the six
doubles, which grant four moves, and the fifteen mixed rolls, which grant two) is expanded with
optimal bear-off play, and results are memoised on the configuration. The reachable configuration
space (at most 15 checkers on 6 points) has ``C(21, 6) = 54264`` states, so the cache stays modest.

From two such one-sided distributions the module computes, **exactly**, the race win probability for
the side on roll -- the probability it bears its last checker off no later than the opponent, with
the on-roll side breaking ties because it moves first -- and a gammon-aware bear-off equity (the
opponent is gammoned when it has borne off nothing by the time the winner finishes).

Scope of exactness. The roll-count DP and the win/gammon probabilities derived from it are exact for
**pure bear-off positions**: every checker of both sides is in its home board (and none on the bar).
For a disengaged position where a side still has checkers outside its home board (a long RACE), an
exact two-sided treatment is far heavier, so :func:`race_win_probability` falls back to a normal
effective-pip-count model; the boundary is documented at the call site and the composite evaluator
relies on the DP only once both sides are home. This module is pure Python and never imports torch.
"""

from functools import cache
from itertools import permutations
import math

from rlgammon.endgame.board_decode import BoardLayout, SideLayout, decode_board, side_layout_for
from rlgammon.endgame.endgame_errors.endgame_errors import InvalidHomeConfigError, NonBearoffPositionError
from rlgammon.endgame.endgame_types import CHECKERS_PER_SIDE, HOME_BOARD_SIZE, HomeConfig, Phase
from rlgammon.endgame.phase import detect_phase_from_layout
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.rlgammon_types import BLACK, MAX_DICE, MIN_DICE, WHITE

# Points won for a single game, a gammon and a backgammon (full-scoring backgammon).
SINGLE_POINTS = 1.0
GAMMON_POINTS = 2.0
# The 21 distinct dice rolls as (high, low) with the probability mass of each (out of 36 ordered
# outcomes): a double (i == j) occurs once, a mixed roll (i != j) occurs twice.
_DICE_ROLLS: tuple[tuple[int, int, float], ...] = tuple(
    (high, low, (1.0 if high == low else 2.0) / 36.0)
    for high in range(MIN_DICE, MAX_DICE + 1)
    for low in range(MIN_DICE, high + 1)
)
# Mean pips cleared per roll for the effective-pip-count race fallback (gnubg's classic EPC constant:
# the long-run average roll bears off roughly this many pips with sensible wastage).
MEAN_PIPS_PER_ROLL = 8.17
# Standard deviation of pips per roll, used to model the race as a difference of normal roll counts.
PIPS_PER_ROLL_STD = 2.0
# Below this many remaining rolls a race-by-pips normal model is meaningless; clamp the spread.
MIN_RACE_ROLLS = 1e-6
# How many standard deviations of roll-count to span when discretising the race normal distribution.
RACE_SIGMA_SPAN = 5.0


def home_config_from_points(points: tuple[int, ...]) -> HomeConfig:
    """
    Validate and normalise six home-point checker counts into a :data:`HomeConfig`.

    :param points: six non-negative counts ordered from the 6-point (first) to the 1-point (last)
    :return: the validated home configuration (6-point first, 1-point last)
    :raises InvalidHomeConfigError: if the length, signs or total (``> 15``) are invalid
    """
    if len(points) != HOME_BOARD_SIZE or any(count < 0 for count in points) or sum(points) > CHECKERS_PER_SIDE:
        raise InvalidHomeConfigError
    return (points[0], points[1], points[2], points[3], points[4], points[5])


def _is_empty(config: HomeConfig) -> bool:
    """
    Return whether a home configuration has no checkers left on the board (all borne off).

    :param config: the home configuration
    :return: ``True`` iff every home point is empty
    """
    return sum(config) == 0


@cache
def _play_single_die(config: HomeConfig, die: int) -> tuple[HomeConfig, ...]:
    """
    Return every home configuration reachable by legally playing a single die in the bear-off.

    Bear-off rules for a die ``die`` (the home points are indexed 0..5 for the 6-point..1-point, so
    the point at pip distance ``p`` is index ``HOME_BOARD_SIZE - p``):

    * a checker on the exact ``die``-point bears off;
    * if the ``die``-point is empty and no checker sits on a higher point, the highest occupied
      point below ``die`` bears off (overshoot is allowed only when nothing is further back);
    * any checker on a point higher than ``die`` may instead move ``die`` pips toward the 1-point
      (a non-bearing move), which is always legal and never bears that checker off.

    A die that cannot be played at all (only possible when the board is empty) yields the unchanged
    configuration.

    :param config: the current home configuration
    :param die: the die pip value (1..6)
    :return: the distinct successor configurations after playing the die
    """
    successors: set[HomeConfig] = set()
    counts = list(config)
    die_index = HOME_BOARD_SIZE - die  # index of the point at pip distance == die
    highest_occupied_index = next((index for index in range(HOME_BOARD_SIZE) if counts[index] > 0), HOME_BOARD_SIZE)
    highest_occupied_pip = HOME_BOARD_SIZE - highest_occupied_index if highest_occupied_index < HOME_BOARD_SIZE else 0

    # Exact bear-off: a checker stands on the die-point.
    if counts[die_index] > 0:
        moved = counts.copy()
        moved[die_index] -= 1
        successors.add(_as_config(moved))

    # Overshoot bear-off: die exceeds the rearmost checker, so the rearmost checker bears off.
    if die > highest_occupied_pip > 0:
        moved = counts.copy()
        moved[highest_occupied_index] -= 1
        successors.add(_as_config(moved))

    # Non-bearing moves: any checker on a point higher than the die slides die pips toward the off.
    for source_index in range(highest_occupied_index, die_index):
        if counts[source_index] > 0:
            moved = counts.copy()
            moved[source_index] -= 1
            moved[source_index + die] += 1
            successors.add(_as_config(moved))

    if not successors:
        return (config,)
    return tuple(successors)


def _as_config(counts: list[int]) -> HomeConfig:
    """
    Pack a length-6 list of home-point counts into a :data:`HomeConfig` tuple.

    :param counts: the six home-point counts (6-point first)
    :return: the equivalent home configuration tuple
    """
    return (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])


def _distinct_orderings(dice: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """
    Return the distinct orderings of a roll's dice (one for a double, the two halves for a mixed roll).

    :param dice: the dice of the roll (length 2 for a mixed roll, length 4 for a double)
    :return: the distinct orderings to expand
    """
    if len(set(dice)) == 1:  # a double: all dice equal, so a single ordering suffices.
        return (dice,)
    return tuple(dict.fromkeys(permutations(dice)))


def _play_roll(config: HomeConfig, dice: tuple[int, ...]) -> frozenset[HomeConfig]:
    """
    Return every configuration reachable by playing a full roll's dice, over every die ordering.

    The two dice of a mixed roll may legally be played in either order, and the order can change
    which checkers can bear off (e.g. with a high and a low die a checker may need the high die
    first to clear), so every distinct permutation of ``dice`` is expanded and the results unioned.
    For a double all four dice are equal and a single ordering suffices. The optimal-play selection
    downstream takes the lowest-expected-rolls successor, which automatically respects backgammon's
    "use as many dice as you can" rule because playing an extra die never raises the pip count.

    :param config: the starting configuration
    :param dice: the dice to play (length 2 for a mixed roll, length 4 for a double)
    :return: the configurations reachable after playing all the dice, over any ordering
    """
    reachable: set[HomeConfig] = set()
    for ordering in _distinct_orderings(dice):
        frontier: set[HomeConfig] = {config}
        for die in ordering:
            frontier = {successor for current in frontier for successor in _play_single_die(current, die)}
        reachable |= frontier
    return frozenset(reachable)


def _distribution_mean(distribution: tuple[float, ...]) -> float:
    """
    Return the mean (expected roll count) of a roll-count distribution.

    :param distribution: a roll-count probability distribution (index ``k`` = ``P(exactly k)``)
    :return: the expected number of rolls
    """
    return math.fsum(rolls * mass for rolls, mass in enumerate(distribution))


@cache
def _mean_rolls(config: HomeConfig) -> float:
    """
    Return the optimal-play expected roll count for a configuration (memoised on the configuration).

    This is the selection key for the optimal successor inside :func:`bearoff_distribution`; caching
    it per configuration keeps the inner ``min`` over a roll's successors cheap.

    :param config: the home configuration
    :return: the expected number of rolls to clear ``config`` under optimal play
    """
    return _distribution_mean(bearoff_distribution(config))


@cache
def bearoff_distribution(config: HomeConfig) -> tuple[float, ...]:
    """
    Return the exact distribution over the number of rolls to bear this side off, under optimal play.

    Entry ``k`` of the returned tuple is the probability that the side needs exactly ``k`` rolls to
    bear all its checkers off, assuming on every roll it plays the dice to minimise the *expected*
    number of remaining rolls. The tuple is indexed from ``0`` and sums to ``1``; index ``0`` carries
    probability ``1`` only for the empty (all-off) configuration.

    The recursion is a single memoised dynamic program: every roll's reachable successors share the
    same already-cached sub-distributions, the optimal successor is the one with the smallest mean,
    and the position's distribution is the probability-weighted mixture of ``1 + successor`` over the
    21 rolls. The state graph is acyclic (every die lowers the pip total), so memoising on the
    configuration is well-defined; :func:`expected_rolls_to_bear_off` reads the mean off the result.

    :param config: the home configuration (validate user input with :func:`home_config_from_points`)
    :return: the roll-count probability distribution (index ``k`` = ``P(exactly k rolls)``)
    """
    if _is_empty(config):
        return (1.0,)
    best_successor_distributions: list[tuple[float, tuple[float, ...]]] = []
    for high, low, probability in _DICE_ROLLS:
        dice = (high,) * 4 if high == low else (high, low)
        best_successor = min(_play_roll(config, dice), key=_mean_rolls)
        best_successor_distributions.append((probability, bearoff_distribution(best_successor)))
    longest = max(len(distribution) for _, distribution in best_successor_distributions)
    # Each branch contributes 1 + (rolls drawn from the successor), so shift its distribution right by one.
    result = [0.0] * (longest + 1)
    for probability, distribution in best_successor_distributions:
        for rolls, mass in enumerate(distribution):
            result[rolls + 1] += probability * mass
    return tuple(result)


def expected_rolls_to_bear_off(config: HomeConfig) -> float:
    """
    Return the exact expected number of rolls to bear this side off under optimal play.

    :param config: the home configuration
    :return: the minimised expected number of rolls to clear the board
    """
    return _distribution_mean(bearoff_distribution(config))


def _cumulative_at_most(distribution: tuple[float, ...], rolls: int) -> float:
    """
    Return ``P(rolls needed <= rolls)`` for a roll-count distribution.

    :param distribution: a roll-count probability distribution (index ``k`` = ``P(exactly k)``)
    :param rolls: the roll budget to accumulate up to (inclusive)
    :return: the probability the side finishes within ``rolls`` rolls
    """
    if rolls < 0:
        return 0.0
    return math.fsum(distribution[: rolls + 1])


def _exact_bearoff_win_probability(my_config: HomeConfig, opp_config: HomeConfig, *, on_roll: bool) -> float:
    """
    Return the exact probability the side to move wins the race from two home configurations.

    Both sides draw their number of rolls-to-finish independently from their one-sided
    distributions. The side on roll moves first, so it wins whenever it finishes in no more rolls
    than the opponent (a tie goes to the mover); when it is *not* on roll the opponent moves first,
    so the mover wins only by finishing strictly sooner.

    :param my_config: the side-to-move's home configuration
    :param opp_config: the opponent's home configuration
    :param on_roll: whether the side to move rolls first
    :return: the side-to-move's exact win probability in ``[0, 1]``
    """
    my_distribution = bearoff_distribution(my_config)
    opp_distribution = bearoff_distribution(opp_config)
    win_probability = 0.0
    for my_rolls, my_mass in enumerate(my_distribution):
        if my_mass == 0.0:
            continue
        # On roll: opponent needs >= my_rolls to lose (ties to mover) -> P(opp >= my_rolls).
        # Not on roll: opponent needs > my_rolls to lose -> P(opp >= my_rolls + 1).
        opp_threshold = my_rolls if on_roll else my_rolls + 1
        opponent_loses = 1.0 - _cumulative_at_most(opp_distribution, opp_threshold - 1)
        win_probability += my_mass * opponent_loses
    return min(max(win_probability, 0.0), 1.0)


def _rolls_to_finish_distribution(side: SideLayout) -> tuple[float, ...]:
    """
    Return a roll-count distribution for a side clearing every checker (exact if all home).

    When the side has all checkers home this is the exact one-sided bear-off distribution; otherwise
    the side is still racing and the count is approximated by a normal centred on
    ``pips / MEAN_PIPS_PER_ROLL`` discretised to whole rolls (the documented race fallback).

    :param side: the side's full layout (in its own direction)
    :return: a roll-count probability distribution (index ``k`` = ``P(exactly k rolls)``)
    """
    if side.all_home():
        return bearoff_distribution(home_config_from_points(side.home_config()))
    return _normal_roll_distribution(side.pip_count() / MEAN_PIPS_PER_ROLL)


def _rolls_to_first_bearoff_distribution(side: SideLayout) -> tuple[float, ...]:
    """
    Return a roll-count distribution for a side bearing off its *first* checker.

    A side that has already borne a checker off needs zero rolls. A side with all 15 home bears one
    off on its very next roll (a home-board roll always removes at least one checker), so it needs
    exactly one roll. Otherwise the rearmost checker must first travel into the home board and then
    off, which is approximated by a normal on ``(rearmost_pip) / MEAN_PIPS_PER_ROLL`` rolls.

    :param side: the side's full layout (in its own direction)
    :return: a roll-count probability distribution for clearing the first checker
    """
    if side.off > 0:
        return (1.0,)
    if side.all_home():
        return (0.0, 1.0)
    # The rearmost checker must reach pip distance 0; model its solo journey by pips per roll.
    return _normal_roll_distribution(side.rearmost_pip() / MEAN_PIPS_PER_ROLL)


def _gammon_probability(winner: SideLayout, loser: SideLayout, *, winner_on_roll: bool) -> float:
    """
    Return the probability the winning side gammons the loser (the loser bears off nothing).

    The winner gammons the loser when it clears all 15 checkers before the loser removes a single
    one. Modelling the winner's rolls-to-finish ``W`` and the loser's rolls-to-first-bear-off ``L``
    as independent roll counts, and accounting for who rolls first, the gammon occurs when the
    loser has not yet completed its first bear-off by the time the winner finishes:

    * winner on roll -- after the winner's ``w`` rolls the loser has had ``w - 1`` rolls, so a gammon
      needs ``L > w - 1`` i.e. ``L >= w``;
    * winner not on roll -- the loser rolls first and has had ``w`` rolls, so a gammon needs
      ``L > w`` i.e. ``L >= w + 1``.

    For a pure bear-off (both home) this yields essentially zero -- the loser always clears a checker
    on its first roll -- matching backgammon theory; it becomes meaningful only when the loser is
    still trapped far from home (a RACE), where the rolls-to-first-bear-off model supplies the mass.

    :param winner: the winning side's layout (in its own direction)
    :param loser: the losing side's layout (in its own direction)
    :param winner_on_roll: whether the winning side rolls first
    :return: the gammon probability in ``[0, 1]``
    """
    if loser.off > 0:
        return 0.0
    winner_finish = _rolls_to_finish_distribution(winner)
    loser_first = _rolls_to_first_bearoff_distribution(loser)
    gammon_probability = 0.0
    for winner_rolls, winner_mass in enumerate(winner_finish):
        if winner_mass == 0.0:
            continue
        loser_threshold = winner_rolls if winner_on_roll else winner_rolls + 1
        loser_survives = 1.0 - _cumulative_at_most(loser_first, loser_threshold - 1)
        gammon_probability += winner_mass * loser_survives
    return min(max(gammon_probability, 0.0), 1.0)


def _normal_roll_distribution(mean_rolls: float) -> tuple[float, ...]:
    """
    Return a discrete roll-count distribution from a normal on the expected number of rolls.

    The (continuous) rolls-to-finish is modelled as ``Normal(mean_rolls, sigma^2)`` with the spread
    growing as the square root of the mean (independent per-roll pip noise), then bucketed into whole
    roll counts ``k >= 1`` by integrating the normal over ``[k - 0.5, k + 0.5]``. This backs the
    long-race fallback so a race and a bear-off expose the same roll-count interface.

    :param mean_rolls: the expected number of rolls to finish (``pips / MEAN_PIPS_PER_ROLL``)
    :return: a roll-count probability distribution (index ``k`` = ``P(exactly k rolls)``)
    """
    if mean_rolls <= MIN_RACE_ROLLS:
        return (0.0, 1.0)
    sigma = max(PIPS_PER_ROLL_STD / MEAN_PIPS_PER_ROLL * math.sqrt(mean_rolls), MIN_RACE_ROLLS)
    max_rolls = math.ceil(mean_rolls + RACE_SIGMA_SPAN * sigma) + 1
    distribution = [0.0] * (max_rolls + 1)
    for rolls in range(1, max_rolls + 1):
        lower = (rolls - 0.5 - mean_rolls) / (sigma * math.sqrt(2.0))
        upper = (rolls + 0.5 - mean_rolls) / (sigma * math.sqrt(2.0))
        distribution[rolls] = 0.5 * (math.erf(upper) - math.erf(lower))
    total = math.fsum(distribution)
    if total > 0.0:
        distribution = [mass / total for mass in distribution]
    return tuple(distribution)


def _effective_pip_race_win_probability(my_pips: int, opp_pips: int, *, on_roll: bool) -> float:
    """
    Approximate a long-race win probability for the side to move from the two pip counts.

    Each side's rolls-to-finish is modelled as ``pips / MEAN_PIPS_PER_ROLL`` with a normal spread,
    and the race is the comparison of two independent such roll counts (the mover gets a half-roll
    edge when on roll). This is the documented fallback used only when at least one side still has
    checkers outside its home board, so the exact bear-off DP does not yet apply.

    :param my_pips: the side-to-move's pip count
    :param opp_pips: the opponent's pip count
    :param on_roll: whether the side to move rolls first
    :return: the side-to-move's approximate win probability in ``[0, 1]``
    """
    my_rolls = my_pips / MEAN_PIPS_PER_ROLL
    opp_rolls = opp_pips / MEAN_PIPS_PER_ROLL
    # Being on roll is worth roughly half a roll of advantage to the mover.
    roll_difference = (opp_rolls - my_rolls) + (0.5 if on_roll else -0.5)
    spread = PIPS_PER_ROLL_STD / MEAN_PIPS_PER_ROLL * math.sqrt(max(my_rolls + opp_rolls, MIN_RACE_ROLLS))
    return 0.5 * (1.0 + math.erf(roll_difference / (spread * math.sqrt(2.0))))


def race_win_probability(my_config: SideLayout, opp_config: SideLayout, *, on_roll: bool) -> float:
    """
    Return the side-to-move's win probability for a disengaged (race/bear-off) position.

    When both sides have every checker in their home board the result is the **exact** two-sided
    bear-off win probability (:func:`_exact_bearoff_win_probability`). Otherwise the position is a
    long race and the result is the effective-pip-count approximation
    (:func:`_effective_pip_race_win_probability`); this boundary is where exact bear-off play takes
    over from the pip model.

    :param my_config: the side-to-move's full side layout (in its own direction)
    :param opp_config: the opponent's full side layout (in its own direction)
    :param on_roll: whether the side to move rolls first
    :return: the side-to-move's win probability in ``[0, 1]``
    """
    if my_config.all_home() and opp_config.all_home():
        return _exact_bearoff_win_probability(
            home_config_from_points(my_config.home_config()),
            home_config_from_points(opp_config.home_config()),
            on_roll=on_roll,
        )
    return _effective_pip_race_win_probability(my_config.pip_count(), opp_config.pip_count(), on_roll=on_roll)


def _equity_from_layout(layout: BoardLayout, perspective: int, *, on_roll: bool) -> float:
    """
    Return ``perspective``'s gammon-aware race/bear-off equity from a decoded layout.

    The equity is ``P(win) * value(win) - P(lose) * value(lose)`` where a win/loss is worth two
    points when it is a gammon (the loser has borne off nothing) and one point otherwise. Gammon
    masses are exact only for pure bear-off positions; in the pip-model race fallback they are zero,
    so the equity reduces to ``2 * P(win) - 1`` there.

    :param layout: the decoded board layout
    :param perspective: the side whose equity to return (WHITE=0, BLACK=1)
    :param on_roll: whether ``perspective`` is the side to move (rolls first)
    :return: ``perspective``'s equity in points (within ``[-2, 2]``)
    """
    me = side_layout_for(layout, perspective, decoded_from=perspective)
    opponent = side_layout_for(layout, 1 - perspective, decoded_from=perspective)
    win_probability = race_win_probability(me, opponent, on_roll=on_roll)
    lose_probability = 1.0 - win_probability
    # I gammon the opponent when I clear out before it bears a checker off; it gammons me with the
    # winner/loser roles and the on-roll flag reversed. Both reduce to ~0 in a pure bear-off. A
    # gammon is a subset of the corresponding win, so clamp each below its win/lose probability.
    my_gammon = min(_gammon_probability(me, opponent, winner_on_roll=on_roll), win_probability)
    opp_gammon = min(_gammon_probability(opponent, me, winner_on_roll=not on_roll), lose_probability)
    win_equity = (win_probability - my_gammon) * SINGLE_POINTS + my_gammon * GAMMON_POINTS
    lose_equity = (lose_probability - opp_gammon) * SINGLE_POINTS + opp_gammon * GAMMON_POINTS
    return win_equity - lose_equity


def bearoff_equity(state: GameState, perspective: int) -> float:
    """
    Return ``perspective``'s exact/near-exact equity for a race or bear-off ``state``.

    The position is decoded from ``perspective``'s observation tensor and must be disengaged (RACE or
    BEAROFF); the equity is exact and gammon-aware for pure bear-off positions and uses the
    effective-pip-count race model otherwise. The side to move is read from ``state.current_player``
    so the half-roll on-roll advantage is attributed correctly.

    :param state: the disengaged game state to evaluate
    :param perspective: the side whose equity to return (WHITE=0, BLACK=1)
    :return: ``perspective``'s equity in points
    :raises NonBearoffPositionError: if the position still has contact (it is not a race/bear-off)
    """
    layout = decode_board(state, perspective)
    if detect_phase_from_layout(layout) == Phase.CONTACT:
        raise NonBearoffPositionError
    mover = state.current_player()
    on_roll = mover in (WHITE, BLACK) and mover == perspective
    return _equity_from_layout(layout, perspective, on_roll=on_roll)
