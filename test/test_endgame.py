"""Tests for the endgame package: board decode, phase detection, exact bear-off DP, composite routing.

The bear-off dynamic program is checked against an *independent* brute-force enumeration of dice
sequences with exhaustive optimal play on small positions; the race/win-probability and phase logic
are checked on hand-built layouts and (when OpenSpiel is installed) on real backgammon states.
"""

import itertools
import math

import numpy as np
import pytest

from rlgammon.endgame.bearoff import (
    MEAN_PIPS_PER_ROLL,
    bearoff_distribution,
    bearoff_equity,
    expected_rolls_to_bear_off,
    home_config_from_points,
    race_win_probability,
)
from rlgammon.endgame.board_decode import (
    FLOATS_PER_POINT,
    NUM_POINTS,
    OBSERVATION_TENSOR_LENGTH,
    OPPONENT_BLOCK_OFFSET,
    OPPONENT_OFF_OFFSET,
    OWN_OFF_OFFSET,
    SideLayout,
    decode_board,
)
from rlgammon.endgame.composite_evaluator import CompositeEvaluator
from rlgammon.endgame.endgame_errors.endgame_errors import InvalidHomeConfigError, NonBearoffPositionError
from rlgammon.endgame.endgame_types import CHECKERS_PER_SIDE, HOME_BOARD_SIZE, Phase
from rlgammon.endgame.phase import detect_phase, detect_phase_from_layout
from rlgammon.game import PossibleEngine, apply_sampled_chance, create_game
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.openspiel_adapter import is_openspiel_available
from rlgammon.rlgammon_types import BLACK, WHITE

# Numerical tolerances.
EXACT_TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 1e-9
# A near-certain win probability threshold for a decisive lead.
DECISIVE_WIN = 0.99
# The win probability of a dead-even race with no on-roll edge (the baseline to beat when on roll).
EVEN_RACE_BASELINE = 0.5
# Standard opening pip count per side.
OPENING_PIP_COUNT = 167
# Bounds for the random-game searches in the real-OpenSpiel tests.
MAX_SEARCH_GAMES = 200
MAX_SEARCH_PLIES = 4000
NUM_SAMPLE_GAMES = 20
# A trivial constant equity returned by the fake contact evaluator used to detect routing.
CONTACT_SENTINEL_EQUITY = 0.123
# The largest equity magnitude the race specialist can return (single game or gammon, no backgammon).
SPECIALIST_EQUITY_BOUND = 2.0
# All 36 ordered dice rolls used by the brute-force reference.
_ORDERED_ROLLS = tuple((high, low) for high in range(1, HOME_BOARD_SIZE + 1) for low in range(1, HOME_BOARD_SIZE + 1))


# --------------------------------------------------------------------------------------------------
# Independent brute-force reference for the one-sided bear-off (small positions only).
# --------------------------------------------------------------------------------------------------
def _brute_min_rolls_distribution(config: tuple[int, ...], max_rolls: int) -> tuple[float, ...]:
    """
    Compute the roll-count distribution of a small bear-off by exhaustive optimal play.

    This deliberately re-derives the answer without the package's DP: it expands all 36 *ordered*
    dice rolls (averaging over them, so doubles automatically carry weight ``1/36`` and mixed rolls
    ``2/36``), plays every die exhaustively (bearing off or moving within the board) via its own move
    generator, and selects per roll the successor minimising the expected remaining rolls. The state
    graph is acyclic -- every die strictly lowers the pip total -- so a plain memo on the state is
    well-defined with no depth truncation. It is tractable only for tiny configurations, which is
    exactly where an independent check is most valuable. ``max_rolls`` only sizes the assertion loop.

    :param config: the home configuration (6-point first, 1-point last)
    :param max_rolls: an (unused-by-the-recursion) hint kept for symmetry with the test's bound
    :return: the roll-count distribution under optimal play (index ``k`` = ``P(exactly k rolls)``)
    """
    del max_rolls  # the recursion terminates structurally; the parameter only documents the bound.
    expectation_memo: dict[tuple[int, ...], float] = {}
    distribution_memo: dict[tuple[int, ...], tuple[float, ...]] = {}

    def expected(state: tuple[int, ...]) -> float:
        if sum(state) == 0:
            return 0.0
        if state in expectation_memo:
            return expectation_memo[state]
        total = 0.0
        for high, low in _ORDERED_ROLLS:
            dice = (high,) * 4 if high == low else (high, low)
            total += (1.0 + min(expected(successor) for successor in _reachable(state, dice))) / len(_ORDERED_ROLLS)
        expectation_memo[state] = total
        return total

    def distribution(state: tuple[int, ...]) -> tuple[float, ...]:
        if sum(state) == 0:
            return (1.0,)
        if state in distribution_memo:
            return distribution_memo[state]
        accumulated = [0.0]
        for high, low in _ORDERED_ROLLS:
            dice = (high,) * 4 if high == low else (high, low)
            best_successor = min(_reachable(state, dice), key=expected)
            shifted = [0.0, *distribution(best_successor)]
            if len(shifted) > len(accumulated):
                accumulated.extend([0.0] * (len(shifted) - len(accumulated)))
            for index, mass in enumerate(shifted):
                accumulated[index] += mass / len(_ORDERED_ROLLS)
        result = tuple(accumulated)
        distribution_memo[state] = result
        return result

    return distribution(config)


def _reachable(state: tuple[int, ...], dice: tuple[int, ...]) -> set[tuple[int, ...]]:
    """
    Return every configuration reachable from ``state`` by playing all ``dice`` over every ordering.

    The dice of a roll may be played in either order and the order can matter, so every distinct
    permutation is expanded and the results unioned (the independent analogue of the package's own
    order-complete roll expansion).

    :param state: the starting configuration (6-point first)
    :param dice: the dice of the roll to play in any order
    :return: the set of reachable configurations
    """
    reachable: set[tuple[int, ...]] = set()
    for ordering in set(itertools.permutations(dice)):
        frontier = {state}
        for die in ordering:
            nxt: set[tuple[int, ...]] = set()
            for current in frontier:
                nxt.update(_play_die_brute(current, die))
            frontier = nxt
        reachable |= frontier
    return reachable


def _play_die_brute(state: tuple[int, ...], die: int) -> list[tuple[int, ...]]:
    """
    Return every configuration reachable by playing a single ``die`` (brute-force bear-off rules).

    :param state: the configuration (6-point first)
    :param die: the die value (1..6)
    :return: the list of successor configurations
    """
    counts = list(state)
    successors: set[tuple[int, ...]] = set()
    occupied_pips = [HOME_BOARD_SIZE - index for index, count in enumerate(counts) if count > 0]
    highest = max(occupied_pips, default=0)
    die_index = HOME_BOARD_SIZE - die
    if counts[die_index] > 0:  # exact bear-off
        moved = counts.copy()
        moved[die_index] -= 1
        successors.add(tuple(moved))
    if die > highest > 0:  # overshoot bear-off (nothing further back)
        moved = counts.copy()
        moved[HOME_BOARD_SIZE - highest] -= 1
        successors.add(tuple(moved))
    for pip in range(die + 1, HOME_BOARD_SIZE + 1):  # in-board moves from higher points
        source = HOME_BOARD_SIZE - pip
        if counts[source] > 0:
            moved = counts.copy()
            moved[source] -= 1
            moved[source + die] += 1
            successors.add(tuple(moved))
    return list(successors) if successors else [state]


# --------------------------------------------------------------------------------------------------
# A hand-built fake state to exercise the decode/phase/composite paths without OpenSpiel.
# --------------------------------------------------------------------------------------------------
def _encode_point(count: int) -> list[float]:
    """
    Encode a checker count into its four observation floats (the inverse of the decoder).

    :param count: the number of checkers on the point
    :return: the four-float group ``(a, b, c, d)``
    """
    if count <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    if count == 1:
        return [1.0, 0.0, 0.0, 0.0]
    if count == 2:  # noqa: PLR2004 - the unary prefix is intrinsic to the encoding
        return [0.0, 1.0, 0.0, 0.0]
    if count == 3:  # noqa: PLR2004 - the unary prefix is intrinsic to the encoding
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, float(count - 3)]


def _encode_side_block(points_pip: dict[int, int], physical_player: int) -> list[float]:
    """
    Encode one physical player's 24-point block from a pip-distance -> count mapping.

    :param points_pip: a mapping of pip distance (1..24) to checker count for the side
    :param physical_player: the physical player owning the block (WHITE=0, BLACK=1)
    :return: the 96-float block in that player's native tensor direction
    """
    # In the uniform convention, pip distance p sits at tensor index 24 - p for WHITE; BLACK is the
    # reverse (index p - 1), matching the engine's opposite travelling directions.
    block = [0.0] * (NUM_POINTS * FLOATS_PER_POINT)
    for pip, count in points_pip.items():
        index = (NUM_POINTS - pip) if physical_player == WHITE else (pip - 1)
        block[index * FLOATS_PER_POINT: (index + 1) * FLOATS_PER_POINT] = _encode_point(count)
    return block


class _FakeState:
    """A minimal GameState exposing a hand-built backgammon observation tensor for both perspectives."""

    def __init__(self, white_pips: dict[int, int], black_pips: dict[int, int],
                 white_off: int, black_off: int, mover: int) -> None:
        """
        Construct the fake state from each side's pip->count maps, off counts and the side to move.

        :param white_pips: WHITE's pip-distance -> count map (bar handled by callers as pip 25 omitted)
        :param black_pips: BLACK's pip-distance -> count map
        :param white_off: WHITE's borne-off count
        :param black_off: BLACK's borne-off count
        :param mover: the side to move (WHITE=0, BLACK=1)
        """
        self._white_pips = white_pips
        self._black_pips = black_pips
        self._white_off = white_off
        self._black_off = black_off
        self._mover = mover

    def current_player(self) -> int:
        """Return the side to move (the fake state is always a decision node)."""
        return self._mover

    def is_chance_node(self) -> bool:
        """Return whether a dice roll is pending (never, for the fake decision-node state)."""
        return False

    def is_terminal(self) -> bool:
        """Return whether the game is over (never, for the fake decision-node state)."""
        return False

    def legal_actions(self) -> list[int]:
        """Return a single dummy legal action (the fake state is not used to step play)."""
        return [0]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return no chance outcomes (the fake state is never a chance node)."""
        return []

    def apply_action(self, action: int) -> None:
        """Ignore the action (the fake state is a static position, never stepped)."""

    def returns(self) -> list[float]:
        """Return zero signed returns (the fake state is never terminal)."""
        return [0.0, 0.0]

    def clone(self) -> "_FakeState":
        """Return an independent copy of the fake state."""
        return _FakeState(
            dict(self._white_pips), dict(self._black_pips), self._white_off, self._black_off, self._mover,
        )

    def observation_tensor(self, player: int) -> list[float]:
        """
        Return the length-200 observation tensor from ``player``'s perspective.

        :param player: the perspective player (WHITE=0, BLACK=1)
        :return: the hand-built observation tensor
        """
        own_pips = self._white_pips if player == WHITE else self._black_pips
        opp_pips = self._black_pips if player == WHITE else self._white_pips
        own_off = self._white_off if player == WHITE else self._black_off
        opp_off = self._black_off if player == WHITE else self._white_off
        tensor = [0.0] * OBSERVATION_TENSOR_LENGTH
        tensor[0:OPPONENT_BLOCK_OFFSET] = _encode_side_block(own_pips, player)
        tensor[OPPONENT_BLOCK_OFFSET:NUM_POINTS * FLOATS_PER_POINT * 2] = _encode_side_block(opp_pips, 1 - player)
        tensor[OWN_OFF_OFFSET] = float(own_off)
        tensor[OPPONENT_OFF_OFFSET] = float(opp_off)
        return tensor


def _home_side_layout(config: tuple[int, ...], *, off: int | None = None) -> SideLayout:
    """
    Build a :class:`SideLayout` from a six-point home configuration (6-point first).

    :param config: the home configuration (6-point first, 1-point last)
    :param off: the borne-off count (defaults to ``15 - sum(config)``)
    :return: the side layout with checkers only on the home points
    """
    points = [0] * NUM_POINTS
    for offset, count in enumerate(config):
        pip = HOME_BOARD_SIZE - offset
        points[NUM_POINTS - pip] = count
    borne_off = (15 - sum(config)) if off is None else off
    return SideLayout(points=tuple(points), bar=0, off=borne_off)


# --------------------------------------------------------------------------------------------------
# Bear-off DP correctness.
# --------------------------------------------------------------------------------------------------
def test_single_checker_on_one_point_is_one_roll() -> None:
    """Test that a lone checker on the 1-point always bears off in exactly one roll."""
    distribution = bearoff_distribution(home_config_from_points((0, 0, 0, 0, 0, 1)))
    assert abs(distribution[1] - 1.0) < EXACT_TOLERANCE
    assert abs(expected_rolls_to_bear_off(home_config_from_points((0, 0, 0, 0, 0, 1))) - 1.0) < EXACT_TOLERANCE


def test_single_checker_on_six_point_distribution() -> None:
    """Test the lone-6-point distribution (fails to clear in one roll for exactly 9 of 36 rolls)."""
    distribution = bearoff_distribution(home_config_from_points((1, 0, 0, 0, 0, 0)))
    # A checker on the 6-point clears in one roll unless the roll cannot move it 6 pips: the only
    # failing rolls are the nine combinations whose larger usable progress is < 6 (e.g. {1-x}, {2,1}),
    # giving P(one roll) = 27/36 = 0.75 and P(two rolls) = 0.25.
    assert abs(distribution[1] - 0.75) < EXACT_TOLERANCE
    assert abs(distribution[2] - 0.25) < EXACT_TOLERANCE


@pytest.mark.parametrize(
    "config",
    [
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 2),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 0, 0, 2, 0),
        (0, 0, 0, 0, 1, 2),
        (0, 0, 0, 1, 0, 1),
    ],
)
def test_bearoff_dp_matches_brute_force(config: tuple[int, ...]) -> None:
    """Test the bear-off DP against an independent brute-force enumeration on small positions."""
    dp_distribution = bearoff_distribution(home_config_from_points(config))
    brute_distribution = _brute_min_rolls_distribution(config, max_rolls=8)
    length = max(len(dp_distribution), len(brute_distribution))
    for index in range(length):
        dp_mass = dp_distribution[index] if index < len(dp_distribution) else 0.0
        brute_mass = brute_distribution[index] if index < len(brute_distribution) else 0.0
        assert abs(dp_mass - brute_mass) < EXACT_TOLERANCE, (config, index, dp_mass, brute_mass)


def test_bearoff_distribution_is_a_probability_distribution() -> None:
    """Test that a multi-point bear-off distribution sums to one and matches its own mean."""
    config = home_config_from_points((0, 0, 2, 2, 3, 3))
    distribution = bearoff_distribution(config)
    assert abs(math.fsum(distribution) - 1.0) < PROBABILITY_TOLERANCE
    mean_from_distribution = sum(rolls * mass for rolls, mass in enumerate(distribution))
    assert abs(mean_from_distribution - expected_rolls_to_bear_off(config)) < EXACT_TOLERANCE


def test_home_config_rejects_too_many_checkers() -> None:
    """Test that an over-full home configuration is rejected."""
    with pytest.raises(InvalidHomeConfigError):
        home_config_from_points((15, 1, 0, 0, 0, 0))


# --------------------------------------------------------------------------------------------------
# Race / win-probability sanity.
# --------------------------------------------------------------------------------------------------
def test_symmetric_bearoff_on_roll_advantage() -> None:
    """Test that from identical home boards the on-roll and off-roll win chances complement to one."""
    config = home_config_from_points((2, 2, 2, 2, 2, 2))
    me = _home_side_layout((2, 2, 2, 2, 2, 2))
    opponent = _home_side_layout((2, 2, 2, 2, 2, 2))
    on_roll = race_win_probability(me, opponent, on_roll=True)
    off_roll = race_win_probability(me, opponent, on_roll=False)
    assert abs(on_roll + off_roll - 1.0) < EXACT_TOLERANCE
    assert on_roll > EVEN_RACE_BASELINE  # moving first in a dead-even race is an advantage.
    # The on-roll edge equals half the tie mass of the shared distribution.
    distribution = bearoff_distribution(config)
    tie_mass = math.fsum(mass * mass for mass in distribution)
    assert abs(on_roll - 0.5 * (1.0 + tie_mass)) < EXACT_TOLERANCE


def test_big_lead_is_near_certain_win() -> None:
    """Test that a one-checker-from-off side beats a side that still has 11 checkers to clear."""
    me = _home_side_layout((0, 0, 0, 0, 0, 1))  # one checker on the 1-point: off in a single roll.
    opponent = _home_side_layout((0, 0, 0, 3, 4, 4))  # eleven checkers left (~5 rolls): hopelessly behind.
    assert race_win_probability(me, opponent, on_roll=True) > DECISIVE_WIN
    assert race_win_probability(me, opponent, on_roll=False) > DECISIVE_WIN


def test_exact_bearoff_win_probabilities_complement() -> None:
    """Test that the two sides' win probabilities from any bear-off sum to one (someone must win)."""
    me = _home_side_layout((1, 2, 0, 3, 1, 2))
    opponent = _home_side_layout((0, 1, 4, 0, 2, 1))
    my_win = race_win_probability(me, opponent, on_roll=True)
    opponent_win = race_win_probability(opponent, me, on_roll=False)
    assert abs(my_win + opponent_win - 1.0) < EXACT_TOLERANCE


def test_race_fallback_uses_pip_lead() -> None:
    """Test that with a checker outside home (a RACE), a large pip lead still wins comfortably."""
    me = SideLayout(points=tuple([0] * 18 + [0, 0, 0, 0, 5, 10]), bar=0, off=0)  # all home, 20 pips
    far_back = [0] * NUM_POINTS
    far_back[0] = 8  # eight checkers 24 pips back -> not all home (a long race)
    far_back[NUM_POINTS - 1] = 7
    opponent = SideLayout(points=tuple(far_back), bar=0, off=0)
    assert not opponent.all_home()
    win = race_win_probability(me, opponent, on_roll=True)
    assert win > DECISIVE_WIN
    # Sanity-check the pip model is what answered: the opponent's pip count dwarfs the mover's.
    assert opponent.pip_count() > me.pip_count() + int(MEAN_PIPS_PER_ROLL * 10)


# --------------------------------------------------------------------------------------------------
# Phase detection on hand-built layouts.
# --------------------------------------------------------------------------------------------------
def test_phase_bearoff_when_both_home() -> None:
    """Test that two all-home sides with no contact are classified BEAROFF."""
    state = _FakeState(white_pips={1: 8, 2: 7}, black_pips={1: 9, 3: 6}, white_off=0, black_off=0, mover=WHITE)
    assert detect_phase(state, WHITE) == Phase.BEAROFF
    assert detect_phase(state, BLACK) == Phase.BEAROFF  # phase is perspective-independent.


def test_phase_race_when_disengaged_but_not_home() -> None:
    """Test that disengaged sides with a checker outside home are classified RACE."""
    # WHITE rearmost at pip 8, BLACK rearmost at pip 10 -> 8 + 10 = 18 < 25, so no contact, but a
    # checker outside the home board (pip > 6) keeps it out of BEAROFF.
    state = _FakeState(white_pips={8: 2, 1: 13}, black_pips={10: 1, 2: 14}, white_off=0, black_off=0, mover=WHITE)
    assert detect_phase(state, WHITE) == Phase.RACE
    assert detect_phase(state, BLACK) == Phase.RACE


def test_phase_contact_when_ranges_overlap() -> None:
    """Test that overlapping rearmost checkers (a hit is still possible) are classified CONTACT."""
    # WHITE rearmost at pip 20, BLACK rearmost at pip 20 -> 20 + 20 = 40 >= 25, contact possible.
    state = _FakeState(white_pips={20: 2, 1: 13}, black_pips={20: 2, 1: 13}, white_off=0, black_off=0, mover=WHITE)
    assert detect_phase(state, WHITE) == Phase.CONTACT


def test_decode_round_trips_hand_built_layout() -> None:
    """Test that the decoder recovers the exact checker counts a fake state was built from."""
    state = _FakeState(white_pips={6: 5, 4: 3, 1: 7}, black_pips={5: 4, 2: 11}, white_off=0, black_off=0, mover=WHITE)
    layout = decode_board(state, WHITE)
    assert layout.mover.home_config() == (5, 0, 3, 0, 0, 7)  # 6,5,4,3,2,1-points
    assert layout.mover.pip_count() == 6 * 5 + 4 * 3 + 1 * 7
    # The same physical player decodes identically from the other perspective.
    assert decode_board(state, BLACK).opponent.points == layout.mover.points


# --------------------------------------------------------------------------------------------------
# Composite evaluator routing.
# --------------------------------------------------------------------------------------------------
class _SentinelContactEvaluator:
    """A fake contact evaluator returning a sentinel, so we can detect when the composite routed to it."""

    def __init__(self) -> None:
        """Track whether the evaluator was called."""
        self.called = False

    def evaluate(self, state: GameState, perspective: int) -> float:  # noqa: ARG002 - protocol signature
        """Record the call and return the sentinel equity."""
        self.called = True
        return CONTACT_SENTINEL_EQUITY


def test_composite_routes_bearoff_to_specialist() -> None:
    """Test that a bear-off position bypasses the contact net and is scored by the specialist."""
    contact = _SentinelContactEvaluator()
    composite = CompositeEvaluator(contact)
    # WHITE one checker from off, BLACK still has eleven checkers to clear: a decisive bear-off win.
    state = _FakeState(white_pips={1: 1}, black_pips={3: 3, 2: 4, 1: 4}, white_off=14, black_off=0, mover=WHITE)
    assert composite.phase_of(state, WHITE) == Phase.BEAROFF
    equity = composite.evaluate(state, WHITE)
    assert not contact.called  # routed to the specialist, not the net.
    assert equity > DECISIVE_WIN  # a near-certain win is worth well above +0.99.


def test_composite_routes_contact_to_net() -> None:
    """Test that a contact position is delegated to the provided net evaluator."""
    contact = _SentinelContactEvaluator()
    composite = CompositeEvaluator(contact)
    state = _FakeState(white_pips={20: 2, 1: 13}, black_pips={20: 2, 1: 13}, white_off=0, black_off=0, mover=WHITE)
    assert composite.phase_of(state, WHITE) == Phase.CONTACT
    equity = composite.evaluate(state, WHITE)
    assert contact.called
    assert equity == pytest.approx(CONTACT_SENTINEL_EQUITY)


def test_bearoff_equity_rejects_contact_position() -> None:
    """Test that asking the specialist to score a contact position raises."""
    state = _FakeState(white_pips={20: 2, 1: 13}, black_pips={20: 2, 1: 13}, white_off=0, black_off=0, mover=WHITE)
    with pytest.raises(NonBearoffPositionError):
        bearoff_equity(state, WHITE)


# --------------------------------------------------------------------------------------------------
# Real OpenSpiel states (skipped when pyspiel is unavailable).
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")
def test_opening_is_contact_with_correct_pips() -> None:
    """Test that the real opening position decodes to 167 pips a side and is CONTACT."""
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    apply_sampled_chance(state, np.random.default_rng(0))
    layout = decode_board(state, WHITE)
    assert layout.mover.pip_count() == OPENING_PIP_COUNT
    assert layout.opponent.pip_count() == OPENING_PIP_COUNT
    assert detect_phase(state, WHITE) == Phase.CONTACT
    assert detect_phase(state, BLACK) == Phase.CONTACT


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")
def test_real_states_decode_consistently_across_perspectives() -> None:
    """Test that random real states decode to 15 checkers a side and to a perspective-stable phase."""
    rng = np.random.default_rng(5)
    game = create_game(PossibleEngine.OPEN_SPIEL)
    seen_non_contact = False
    for _ in range(NUM_SAMPLE_GAMES):
        state = game.new_initial_state()
        plies = 0
        while not state.is_terminal() and plies < MAX_SEARCH_PLIES:
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
            else:
                legal = state.legal_actions()
                state.apply_action(int(legal[rng.integers(len(legal))]))
            plies += 1
            if state.is_chance_node() or state.is_terminal():
                continue
            layout = decode_board(state, state.current_player())
            assert sum(layout.mover.points) + layout.mover.bar + layout.mover.off == CHECKERS_PER_SIDE
            assert sum(layout.opponent.points) + layout.opponent.bar + layout.opponent.off == CHECKERS_PER_SIDE
            assert detect_phase(state, WHITE) == detect_phase(state, BLACK)
            if detect_phase_from_layout(layout) != Phase.CONTACT:
                seen_non_contact = True
    assert seen_non_contact  # the random games should reach at least one race/bear-off.


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")
def test_composite_routes_real_opening_to_net() -> None:
    """Test that the composite sends the real opening (a contact position) to the net evaluator."""
    contact = _SentinelContactEvaluator()
    composite = CompositeEvaluator(contact)
    state = create_game(PossibleEngine.OPEN_SPIEL).new_initial_state()
    apply_sampled_chance(state, np.random.default_rng(0))
    composite.evaluate(state, state.current_player())
    assert contact.called


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) not installed")
def test_specialist_scores_a_real_bearoff_position() -> None:
    """Test that the composite scores a real bear-off position with the specialist (net untouched)."""
    rng = np.random.default_rng(11)
    game = create_game(PossibleEngine.OPEN_SPIEL)
    contact = _SentinelContactEvaluator()
    composite = CompositeEvaluator(contact)
    for _ in range(MAX_SEARCH_GAMES):
        state = game.new_initial_state()
        plies = 0
        while not state.is_terminal() and plies < MAX_SEARCH_PLIES:
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
            else:
                legal = state.legal_actions()
                state.apply_action(int(legal[rng.integers(len(legal))]))
            plies += 1
            if state.is_chance_node() or state.is_terminal():
                continue
            if composite.phase_of(state, state.current_player()) == Phase.BEAROFF:
                equity = composite.evaluate(state, state.current_player())
                assert not contact.called  # the specialist handled it.
                assert -SPECIALIST_EQUITY_BOUND <= equity <= SPECIALIST_EQUITY_BOUND
                return
    pytest.skip("no bear-off position reached in the random play-outs")


def test_all_ordered_rolls_cover_every_pair() -> None:
    """Test the brute-force helper's roll table covers all 36 ordered dice outcomes (guards the reference)."""
    assert len(_ORDERED_ROLLS) == len(list(itertools.product(range(1, HOME_BOARD_SIZE + 1), repeat=2)))
