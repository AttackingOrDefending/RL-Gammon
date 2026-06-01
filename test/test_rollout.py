"""Tests for truncated rollouts with control-variate / antithetic variance reduction.

The deterministic parts run on the pure-Python :class:`~rlgammon.game.mock_game.MockGame` and on a
tiny self-contained random-walk game (so the control variate has a strongly-correlated baseline and
the variance reduction is reproducible without OpenSpiel); the OpenSpiel-needing parts (unbiasedness
of the control variate on real backgammon, the rollout refining the static net) are gated on
:func:`~rlgammon.game.is_openspiel_available`.
"""

import numpy as np
import pytest

from rlgammon.agents.td_agent import TDAgent
from rlgammon.cube.cube_types import CubeOwner, CubeState, GameMode, MatchContext
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    create_game,
    is_openspiel_available,
)
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.models.model_errors.model_errors import ValueHeadConfigError
from rlgammon.models.model_types import ValueHead
from rlgammon.models.value_model import TDGammonNet
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator
from rlgammon.planning.planning_types import Evaluator
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.rollout.cubeful_rollout import CubefulRolloutEvaluator
from rlgammon.rollout.rollout import RolloutEvaluator, rollout_equity
from rlgammon.rollout.rollout_errors.rollout_errors import ChanceRootError, RolloutConfigError
from rlgammon.rollout.rollout_types import RolloutConfig

# Sentinels mirroring the engine's chance / terminal player ids for the random-walk game.
WALK_CHANCE_PLAYER = -1
WALK_TERMINAL_PLAYER = -4
# Distance from the origin at which the random walk terminates (a win at +bound, a loss at -bound).
WALK_BOUND = 8
# A generous trial count for the seeded variance-reduction comparison (deterministic given the seed).
VR_TRIALS = 4000
# A small absolute tolerance for the unbiasedness checks (control variate vs plain rollout means).
UNBIASED_TOLERANCE = 0.04
# A near-zero tolerance for an exactly-decided position's rollout equity.
DECIDED_TOLERANCE = 1e-9
# The point value of a single win/loss in the mock and random-walk games.
WIN_POINTS = 1.0
# The number of OpenSpiel self-play positions used in the gated unbiasedness test.
OPENSPIEL_POSITIONS = 2
# A moderate trial count for the (slower) OpenSpiel-gated rollouts.
OPENSPIEL_TRIALS = 400


class ConstantEvaluator:
    """A leaf evaluator returning a fixed equity for every non-terminal state (truncation probe)."""

    def __init__(self, value: float) -> None:
        """
        Construct the constant evaluator.

        :param value: the constant equity returned for any state and perspective
        """
        self._value = value

    def evaluate(self, state: GameState, perspective: int) -> float:  # noqa: ARG002
        """
        Return the configured constant value regardless of the state or perspective.

        :param state: the game state to evaluate (unused)
        :param perspective: the player whose equity to return (unused)
        :return: the configured constant value
        """
        return self._value


class GreedyMaxPolicy:
    """A deterministic policy that always plays the largest legal action id."""

    def choose_move(self, actions: list[int], state: GameState) -> int:  # noqa: ARG002
        """
        Return the largest legal action id.

        :param actions: the legal action ids at ``state``
        :param state: the current decision-node game state (unused)
        :return: the largest legal action id
        """
        return max(actions)


class WalkState:
    """A 1-D random-walk game state: a token steps +-1 each ply until it reaches +-:data:`WALK_BOUND`.

    The walk is driven entirely by chance (a fair +-1 step) with a single trivial legal action at
    each decision node, so a value evaluator reading the signed position is a strong, monotone
    predictor of the eventual win/loss -- exactly the regime where the lookahead control variate
    cuts variance. It satisfies the :class:`~rlgammon.game.backgammon_protocol.GameState` protocol.
    """

    def __init__(self, position: int = 0, *, chance: bool = False) -> None:
        """
        Construct a walk state at a given signed position.

        :param position: the signed token position (a win at ``+WALK_BOUND``, a loss at ``-WALK_BOUND``)
        :param chance: whether a +-1 step is pending (a chance node) rather than the trivial decision
        """
        self._position = position
        self._chance = chance
        self._terminal = abs(position) >= WALK_BOUND

    def current_player(self) -> int:
        """Return WHITE at a decision node, or the chance / terminal sentinel."""
        if self._terminal:
            return WALK_TERMINAL_PLAYER
        return WALK_CHANCE_PLAYER if self._chance else WHITE

    def is_chance_node(self) -> bool:
        """Return whether a +-1 step is pending."""
        return self._chance and not self._terminal

    def is_terminal(self) -> bool:
        """Return whether the walk has reached a boundary."""
        return self._terminal

    def legal_actions(self) -> list[int]:
        """Return the single trivial action at a decision node, else an empty list."""
        return [] if (self._terminal or self._chance) else [0]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return the fair +-1 step distribution at a chance node, else an empty list."""
        return [] if not self.is_chance_node() else [(0, 0.5), (1, 0.5)]

    def apply_action(self, action: int) -> None:
        """
        Apply a +-1 step (at a chance node) or the trivial decision (advancing to the next step).

        :param action: the step direction at a chance node (1 = +1, else -1); ignored at a decision node
        """
        if self._chance:
            self._position += 1 if action == 1 else -1
            self._chance = False
            self._terminal = abs(self._position) >= WALK_BOUND
        else:
            self._chance = True

    def observation_tensor(self, player: int) -> list[float]:
        """
        Return the signed position scaled to ``[-1, 1]`` from a player's perspective.

        :param player: the player whose perspective to encode (WHITE keeps the sign, BLACK flips it)
        :return: a length-one observation tensor holding the (perspective-signed) scaled position
        """
        sign = 1.0 if player == WHITE else -1.0
        return [sign * self._position / WALK_BOUND]

    def returns(self) -> list[float]:
        """Return the per-player signed returns (meaningful at a boundary)."""
        white = WIN_POINTS if self._position >= WALK_BOUND else -WIN_POINTS
        return [white, -white]

    def clone(self) -> "WalkState":
        """Return an independent deep copy of the walk state."""
        clone = WalkState(self._position, chance=self._chance)
        clone._terminal = self._terminal
        return clone


class PositionEvaluator:
    """A leaf evaluator reading a state's signed (scaled) position as its equity (walk-game baseline)."""

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return the perspective-signed scaled position of the walk state.

        :param state: the walk-game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the perspective-signed scaled position in ``[-1, 1]``
        """
        return float(state.observation_tensor(perspective)[0])


def _decided_state(player: int) -> GameState:
    """
    Build a MockGame decision node that wins immediately for ``player`` (an exactly-decided position).

    :param player: the player to move and win (WHITE=0, BLACK=1)
    :return: the contrived win-in-one decision node
    """
    return MockGame.contrived_win_in_one(player)


def test_rollout_of_decided_position_returns_correct_sign() -> None:
    """Test a win-in-one position rolls out to exactly the winner's single-point return for both sides."""
    policy = GreedyMaxPolicy()
    evaluator = ConstantEvaluator(0.0)
    config = RolloutConfig(num_trials=16, max_depth=4, seed=0, variance_reduction=False)
    for player in (WHITE, BLACK):
        result = rollout_equity(_decided_state(player), evaluator, policy,
                                np.random.default_rng(0), config, perspective=player)
        assert result.equity == pytest.approx(WIN_POINTS, abs=DECIDED_TOLERANCE)
        opp_result = rollout_equity(_decided_state(player), evaluator, policy,
                                    np.random.default_rng(0), config,
                                    perspective=WHITE if player == BLACK else BLACK)
        assert opp_result.equity == pytest.approx(-WIN_POINTS, abs=DECIDED_TOLERANCE)


def test_truncation_bootstraps_with_evaluator() -> None:
    """Test that a depth-1 truncation that never terminates bootstraps with the evaluator's value."""
    # From the initial WALK decision node a single decision ply (max_depth=1) leaves a non-terminal
    # chance node, so every trial must bootstrap the constant evaluator rather than read a return.
    bootstrap_value = 0.25
    config = RolloutConfig(num_trials=8, max_depth=1, seed=0, variance_reduction=False,
                           control_variate_depth=1)
    result = rollout_equity(WalkState(0), ConstantEvaluator(bootstrap_value), GreedyMaxPolicy(),
                            np.random.default_rng(1), config, perspective=WHITE)
    assert result.equity == pytest.approx(bootstrap_value, abs=DECIDED_TOLERANCE)
    assert result.std_error == pytest.approx(0.0, abs=DECIDED_TOLERANCE)


def test_variance_reduction_lowers_std_error_at_equal_trials() -> None:
    """Test that the control-variate rollout has a strictly smaller std-error than plain at equal trials."""
    evaluator = PositionEvaluator()
    policy = GreedyMaxPolicy()
    plain = RolloutConfig(num_trials=VR_TRIALS, max_depth=WALK_BOUND, seed=7, variance_reduction=False)
    reduced = RolloutConfig(num_trials=VR_TRIALS, max_depth=WALK_BOUND, seed=7, variance_reduction=True)
    plain_result = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(3), plain,
                                  perspective=WHITE)
    reduced_result = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(3), reduced,
                                    perspective=WHITE)
    assert reduced_result.std_error < plain_result.std_error
    assert reduced_result.variance_reduced
    # The control variate is unbiased: both estimators agree within their (small) error bars.
    assert reduced_result.equity == pytest.approx(plain_result.equity, abs=UNBIASED_TOLERANCE)


def test_antithetic_composes_with_control_variate() -> None:
    """Test that adding antithetic dice keeps the estimator unbiased and reduces std-error vs plain."""
    evaluator = PositionEvaluator()
    policy = GreedyMaxPolicy()
    plain = RolloutConfig(num_trials=VR_TRIALS, max_depth=WALK_BOUND, seed=5, variance_reduction=False)
    both = RolloutConfig(num_trials=VR_TRIALS // 2, max_depth=WALK_BOUND, seed=5,
                         variance_reduction=True, antithetic=True)
    plain_result = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(9), plain,
                                  perspective=WHITE)
    both_result = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(9), both,
                                 perspective=WHITE)
    assert both_result.std_error < plain_result.std_error
    assert both_result.equity == pytest.approx(plain_result.equity, abs=UNBIASED_TOLERANCE)


def test_common_random_numbers_are_reproducible() -> None:
    """Test that two rollouts sharing a config and seed produce identical estimates (common randoms)."""
    evaluator = PositionEvaluator()
    policy = GreedyMaxPolicy()
    config = RolloutConfig(num_trials=128, max_depth=WALK_BOUND, seed=123, variance_reduction=True)
    first = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(0), config,
                           perspective=WHITE)
    second = rollout_equity(WalkState(0), evaluator, policy, np.random.default_rng(0), config,
                            perspective=WHITE)
    assert first.equity == second.equity
    assert first.std_error == second.std_error


def test_rollout_evaluator_satisfies_evaluator_protocol() -> None:
    """Test that the rollout-backed evaluator structurally satisfies the planning Evaluator protocol."""
    config = RolloutConfig(num_trials=8, max_depth=2, seed=0, variance_reduction=True)
    evaluator = RolloutEvaluator(ConstantEvaluator(0.0), config, policy=GreedyMaxPolicy())
    assert isinstance(evaluator, Evaluator)
    value = evaluator.evaluate(_decided_state(WHITE), WHITE)
    assert isinstance(value, float)


def test_rollout_evaluator_default_policy_picks_winning_afterstate() -> None:
    """Test the rollout evaluator's default 1-ply argmax policy values a win-in-one near a full point."""
    evaluator = RolloutEvaluator(ConstantEvaluator(0.0),
                                 RolloutConfig(num_trials=8, max_depth=3, seed=0))
    # The default policy is a 1-ply argmax of the leaf evaluator: from the win-in-one position it
    # finds the immediately-winning action, so the rollout equity is the full win for the mover.
    result = evaluator.rollout(_decided_state(WHITE), WHITE)
    assert result.equity == pytest.approx(WIN_POINTS, abs=DECIDED_TOLERANCE)


def test_invalid_config_raises() -> None:
    """Test that non-positive trials / depth or an out-of-range control-variate depth are rejected."""
    policy = GreedyMaxPolicy()
    evaluator = ConstantEvaluator(0.0)
    for bad in (RolloutConfig(num_trials=0, max_depth=2),
                RolloutConfig(num_trials=4, max_depth=0),
                RolloutConfig(num_trials=4, max_depth=2, control_variate_depth=3)):
        with pytest.raises(RolloutConfigError):
            rollout_equity(_decided_state(WHITE), evaluator, policy, np.random.default_rng(0), bad,
                           perspective=WHITE)


def test_chance_root_without_perspective_raises() -> None:
    """Test that a chance-node rollout with no explicit perspective raises ``ChanceRootError``."""
    chance_root = MockGame().new_initial_state()
    assert chance_root.is_chance_node()
    with pytest.raises(ChanceRootError):
        rollout_equity(chance_root, ConstantEvaluator(0.0), GreedyMaxPolicy(),
                       np.random.default_rng(0), RolloutConfig(num_trials=4, max_depth=2))


def test_chance_root_with_perspective_is_allowed() -> None:
    """Test that a chance-node root (a move's afterstate) rolls out when a perspective is given."""
    # The opening MockGame state is a chance node; with an explicit perspective the rollout resolves
    # the pending dice like any mid-trial chance node instead of rejecting the root.
    chance_root = MockGame().new_initial_state()
    result = rollout_equity(chance_root, ConstantEvaluator(0.0), GreedyMaxPolicy(),
                            np.random.default_rng(0), RolloutConfig(num_trials=8, max_depth=3),
                            perspective=WHITE)
    assert isinstance(result.equity, float)


def test_cubeful_rollout_evaluator_satisfies_protocol_and_signs() -> None:
    """Test that the cubeful rollout evaluator satisfies the protocol and signs a decided position."""
    model = TDGammonNet(value_head=ValueHead.EQUITY_SIGMOID, seed=1)
    config = RolloutConfig(num_trials=16, max_depth=3, seed=0)
    evaluator = CubefulRolloutEvaluator(model, config, CubeState(owner=CubeOwner.CENTERED),
                                        MatchContext(mode=GameMode.MONEY))
    assert isinstance(evaluator, Evaluator)
    # A win-in-one position is a near-certain win, so the per-point cubeful equity is strongly positive
    # for the winner and strongly negative for the loser (bounded by the +-3 money range).
    max_equity = 3.0
    white_value = evaluator.evaluate(_decided_state(WHITE), WHITE)
    assert 0.0 < white_value <= max_equity
    assert evaluator.evaluate(_decided_state(WHITE), BLACK) < 0.0


def test_cubeful_rollout_requires_equity_head() -> None:
    """Test that the cubeful rollout evaluator rejects a non-EQUITY_SIGMOID value network."""
    scalar_model = TDGammonNet(value_head=ValueHead.SCALAR_TANH, seed=1)
    with pytest.raises(ValueHeadConfigError):
        CubefulRolloutEvaluator(scalar_model, RolloutConfig(num_trials=4, max_depth=2),
                                CubeState(), MatchContext())


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is not installed")
def test_openspiel_control_variate_is_unbiased() -> None:
    """Test that on real backgammon the control-variate rollout matches the plain rollout in mean."""
    agent = TDAgent()
    evaluator = ValueNetEvaluator(agent.get_model())
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(0)
    state = game.new_initial_state()
    while state.is_chance_node():
        apply_sampled_chance(state, rng)
    plain = RolloutConfig(num_trials=OPENSPIEL_TRIALS, max_depth=6, seed=1, variance_reduction=False)
    reduced = RolloutConfig(num_trials=OPENSPIEL_TRIALS, max_depth=6, seed=1, variance_reduction=True)
    plain_result = rollout_equity(state, evaluator, agent, np.random.default_rng(2), plain, perspective=WHITE)
    reduced_result = rollout_equity(state, evaluator, agent, np.random.default_rng(2), reduced,
                                    perspective=WHITE)
    # Unbiased: the two means agree within a few combined standard errors.
    tolerance = 6.0 * (plain_result.std_error + reduced_result.std_error)
    assert reduced_result.equity == pytest.approx(plain_result.equity, abs=tolerance)


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is not installed")
def test_openspiel_rollout_evaluator_returns_finite_equity() -> None:
    """Test that the rollout evaluator returns a finite, in-range equity on a real backgammon position."""
    agent = TDAgent()
    leaf = ValueNetEvaluator(agent.get_model())
    rollout_evaluator = RolloutEvaluator(leaf, RolloutConfig(num_trials=64, max_depth=4, seed=0))
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(0)
    state = game.new_initial_state()
    while state.is_chance_node():
        apply_sampled_chance(state, rng)
    equity = rollout_evaluator.evaluate(state, WHITE)
    assert isinstance(equity, float)
    max_equity = 3.0
    assert -max_equity <= equity <= max_equity
