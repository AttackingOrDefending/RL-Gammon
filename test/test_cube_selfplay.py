"""Smoke and mechanism tests for the cube-testing harness and the cube self-play script.

The harness mechanics (cube-scaled scoring, pass awards, the ``use_cube=False`` regression guard) are
exercised deterministically on the pyspiel-free mock game with scripted cube agents; a short
OpenSpiel run then proves a money cube game plays end-to-end with real doubles.
"""

import numpy as np
import pytest

from rlgammon.agents.td_agent import TDAgent
from rlgammon.cube.cube_errors.cube_errors import CubelessModelError
from rlgammon.cube.cube_types import CubeOwner, CubeState, MatchContext
from rlgammon.cube.met import MET
from rlgammon.game import PossibleEngine, is_openspiel_available
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.models.model_types import ValueHead
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.cube_testing import CubeTesting, NoDoubleMoneyTaker
from scripts.cube_selfplay import evaluate, train_agent

# Number of mock games per harness assertion.
MOCK_GAMES = 8
# The expected points for a single mock win scaled by a taken (doubled) cube.
DOUBLED_SINGLE_POINTS = 2.0
# The expected points awarded when a double is passed on a centred 1-cube.
PASS_POINTS = 1.0
# A short OpenSpiel run proving doubles happen end-to-end.
OPENSPIEL_TRAIN_EPISODES = 60
OPENSPIEL_EVAL_GAMES = 20


class _ScriptedAgent:
    """A deterministic move policy with configurable, scripted cube decisions for tests."""

    def __init__(self, *, doubles: bool, takes: bool) -> None:
        """
        Construct the scripted agent.

        :param doubles: whether the agent always offers a double when it may
        :param takes: whether the agent always takes an offered double
        """
        self._doubles = doubles
        self._takes = takes

    def choose_move(self, actions: list[int], state: GameState) -> int:  # noqa: ARG002
        """
        Return the first legal action (a deterministic, game-agnostic policy).

        :param actions: the legal action ids at the current decision node
        :param state: the current decision-node game state (unused)
        :return: the first legal action id
        """
        return actions[0]

    def should_double(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,  # noqa: ARG002
                      met: MET | None = None, x: float = 0.68) -> bool:  # noqa: ARG002
        """
        Return the scripted doubling decision.

        :param state: the decision-node state (unused)
        :param cube: the current cube state (unused)
        :param match_ctx: the match context (unused)
        :param met: an optional match-equity table (unused)
        :param x: the cube-life index (unused)
        :return: the configured constant doubling decision
        """
        return self._doubles

    def should_take(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,  # noqa: ARG002
                    met: MET | None = None, x: float = 0.68) -> bool:  # noqa: ARG002
        """
        Return the scripted take decision.

        :param state: the decision-node state (unused)
        :param cube: the pre-double cube state (unused)
        :param match_ctx: the match context (unused)
        :param met: an optional match-equity table (unused)
        :param x: the cube-life index (unused)
        :return: the configured constant take decision
        """
        return self._takes


class _TerminalGammon:
    """A minimal terminal state returning a WHITE gammon (+2 / -2) for the Jacoby-clamp test."""

    def returns(self) -> list[float]:
        """Return a WHITE gammon result."""
        return [2.0, -2.0]


def _mock_harness() -> CubeTesting:
    """
    Return a cube-testing harness wired to the pyspiel-free mock game.

    :return: a harness playing on the mock engine
    """
    return CubeTesting(engine=PossibleEngine.MOCK)


def test_money_game_scores_doubled_cube() -> None:
    """Test that a taken double scales the mock single-win score to two points.

    The mock game's first mover (the WHITE seat) always wins under the deterministic move policy, so
    a single game (no colour swap) is used to read the winner's exact cube-scaled points.
    """
    doubler = _ScriptedAgent(doubles=True, takes=True)
    taker = _ScriptedAgent(doubles=False, takes=True)
    harness = _mock_harness()
    result = harness.play_money_games({WHITE: doubler, BLACK: taker}, 1,
                                      np.random.default_rng(0), use_cube=True)
    assert result["doubles"] == pytest.approx(1.0)
    assert result["takes"] == pytest.approx(1.0)
    assert result["win_rate"] == pytest.approx(1.0)
    # WHITE wins a single game on a taken 2-cube, so it scores exactly the doubled stake.
    assert result["ppg"] == pytest.approx(DOUBLED_SINGLE_POINTS)


def test_pass_awards_current_cube_value() -> None:
    """Test that passing a double awards exactly the (pre-double) cube value with no game played."""
    doubler = _ScriptedAgent(doubles=True, takes=False)
    passer = _ScriptedAgent(doubles=False, takes=False)
    harness = _mock_harness()
    # The scored agent always doubles and its opponent always passes, so it wins 1 point per game.
    result = harness.play_money_games({WHITE: doubler, BLACK: passer}, MOCK_GAMES,
                                      np.random.default_rng(1), use_cube=True)
    assert result["passes"] == pytest.approx(1.0)
    assert result["win_rate"] == pytest.approx(1.0)
    assert result["ppg"] == pytest.approx(PASS_POINTS)


def test_use_cube_false_matches_cubeless() -> None:
    """Test that disabling the cube reproduces the cubeless single-point score (regression guard)."""
    doubler = _ScriptedAgent(doubles=True, takes=True)
    # The taker never redoubles, so the cube is turned exactly once and settles at two.
    taker = _ScriptedAgent(doubles=False, takes=True)
    harness = _mock_harness()
    with_cube = harness.play_money_games({WHITE: doubler, BLACK: taker}, 1,
                                         np.random.default_rng(2), use_cube=True)
    no_cube = harness.play_money_games({WHITE: doubler, BLACK: taker}, 1,
                                       np.random.default_rng(2), use_cube=False)
    assert no_cube["doubles"] == pytest.approx(0.0)
    assert no_cube["mean_cube_turns"] == pytest.approx(0.0)
    # The cubeless game is worth a single point; the same game with a taken cube is worth two.
    assert no_cube["ppg"] == pytest.approx(PASS_POINTS)
    assert with_cube["ppg"] == pytest.approx(DOUBLED_SINGLE_POINTS)


def test_mutual_redoubling_reaches_cube_ceiling() -> None:
    """Test that two always-redoubling agents drive the cube up to (but not past) its ceiling."""
    aggressive = _ScriptedAgent(doubles=True, takes=True)
    harness = _mock_harness()
    result = harness.play_money_games({WHITE: aggressive, BLACK: aggressive}, 1,
                                      np.random.default_rng(2), use_cube=True)
    # The winner's score is the cube ceiling; ownership flips on each take so both sides redouble.
    assert abs(result["ppg"]) == pytest.approx(float(CubeState().max_cube))
    assert result["doubles"] > 1


def test_no_double_baseline_never_doubles() -> None:
    """Test that the never-double / always-take baseline produces no doubles when it is on roll."""
    mover = _ScriptedAgent(doubles=False, takes=True)
    baseline = NoDoubleMoneyTaker(mover)
    harness = _mock_harness()
    result = harness.play_money_games({WHITE: baseline, BLACK: baseline}, 1,
                                      np.random.default_rng(3), use_cube=True)
    assert result["doubles"] == pytest.approx(0.0)
    assert result["ppg"] == pytest.approx(PASS_POINTS)


def test_jacoby_clamp_collapses_gammon_on_undoubled_cube() -> None:
    """Test that the live Jacoby rule scores an undoubled win as a single point in the harness."""
    harness = _mock_harness()
    cube = CubeState(value=1, owner=CubeOwner.CENTERED, jacoby=True)
    terminal: GameState = _TerminalGammon()  # type: ignore[assignment]
    # No one doubles, so the cube stays centred and Jacoby clamps any multiplier to a single point.
    scored = harness._score_terminal(terminal, cube, use_cube=True)
    assert scored == (WHITE, PASS_POINTS)


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is not installed")
def test_openspiel_cube_game_plays_end_to_end_with_doubles() -> None:
    """Test a real OpenSpiel money cube game plays end-to-end with doubles and cube-scaled scoring.

    Scripted doublers drive the cube deterministically (independent of the value network's
    calibration) to prove the harness mechanics integrate with the real backgammon engine: a double
    is offered and taken, and the terminal score is the cube-scaled result.
    """
    doubler = _ScriptedAgent(doubles=True, takes=True)
    taker = _ScriptedAgent(doubles=False, takes=True)
    harness = CubeTesting(engine=PossibleEngine.OPEN_SPIEL)
    result = harness.play_money_games({WHITE: doubler, BLACK: taker}, OPENSPIEL_EVAL_GAMES,
                                      np.random.default_rng(0), use_cube=True)
    assert result["games"] == pytest.approx(float(OPENSPIEL_EVAL_GAMES))
    assert result["doubles"] > 0
    assert result["takes"] == pytest.approx(1.0)
    assert result["mean_cube_turns"] > 0

    # A single game (no colour swap) is decided on a taken 2-cube: the WHITE doubler scores >= 2.
    single = harness.play_money_games({WHITE: doubler, BLACK: taker}, 1,
                                      np.random.default_rng(0), use_cube=True)
    assert abs(single["ppg"]) >= DOUBLED_SINGLE_POINTS


@pytest.mark.skipif(not is_openspiel_available(), reason="OpenSpiel (pyspiel) is not installed")
def test_openspiel_trained_agent_eval_runs() -> None:
    """Test that the trained-agent cube self-play evaluation path runs end-to-end on OpenSpiel."""
    agent = train_agent(episodes=OPENSPIEL_TRAIN_EPISODES, hidden=32, lr=0.1, lamda=0.7, seed=0)
    result = evaluate(agent, games=OPENSPIEL_EVAL_GAMES, match_length=0, seed=0, use_cube=True,
                      eval_baseline=True)
    assert result["games"] == pytest.approx(float(OPENSPIEL_EVAL_GAMES))
    # The cube decisions are exercised at every turn; doubles may or may not fire at this tiny scale.
    assert result["mean_cube_turns"] > 0


def test_td_agent_position_probs_requires_equity_head() -> None:
    """Test that position_probs raises on a non-equity-sigmoid model (the cubeless guard)."""
    agent = TDAgent(value_head=ValueHead.SCALAR_TANH)
    state = MockGame.contrived_win_in_one(WHITE)
    with pytest.raises(CubelessModelError):
        agent.position_probs(state, WHITE)
