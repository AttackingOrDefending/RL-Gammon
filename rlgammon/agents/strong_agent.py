"""The strongest near-term backgammon agent, assembled from the existing world-class pieces.

This module is pure *composition*: it wires together components that already exist and are tested
elsewhere into the single strongest play configuration the project currently offers, exposed as one
configurable factory. The stack is:

* **value** -- a calibrated :class:`~rlgammon.models.value_model.TDGammonNet` (loaded through a
  :class:`~rlgammon.agents.td_agent.TDAgent`), whose win / gammon / backgammon components are grounded
  so the cube layer below can use them;
* **leaf evaluator** -- a :class:`~rlgammon.endgame.composite_evaluator.CompositeEvaluator` wrapping the
  net's :class:`~rlgammon.planning.leaf_evaluator.ValueNetEvaluator`, so disengaged (RACE / BEAROFF)
  leaves are scored by the *exact* analytic bear-off specialist and only CONTACT leaves hit the net;
* **search** -- a :class:`~rlgammon.planning.expectiminimax.StarMinimax` expectiminimax (2-ply by
  default, with star2 chance-node pruning), so the agent "thinks deeper at test time" than the 1-ply
  greedy training policy. The agent's :meth:`~rlgammon.agents.planning_agent.PlanningAgent.choose_move`
  returns ``planner.search(state).best_action``;
* optionally a (slower, stronger) **truncated-rollout** leaf evaluator
  (:class:`~rlgammon.rollout.RolloutEvaluator`) wrapping the composite, and an optional **doubling-cube**
  layer that delegates the cube decisions to the same calibrated net.

The factory deliberately reuses :class:`~rlgammon.agents.planning_agent.PlanningAgent`'s search-driven
``choose_move`` rather than re-implementing search; the returned :class:`StrongAgent` is a thin
``PlanningAgent`` subtype that merely *adds* the optional cube methods, so it drops into the existing
agent / evaluation interfaces unchanged.
"""

from dataclasses import dataclass

import numpy as np

from rlgammon.agents.planning_agent import PlanningAgent
from rlgammon.agents.td_agent import TDAgent
from rlgammon.cube.cube_equity import CubeAction
from rlgammon.cube.cube_types import CubeState, MatchContext
from rlgammon.endgame.composite_evaluator import CompositeEvaluator
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.base_search import BaseSearch
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator
from rlgammon.planning.planning_types import Evaluator
from rlgammon.rlgammon_types import WHITE
from rlgammon.rollout.rollout import RolloutEvaluator
from rlgammon.rollout.rollout_types import RolloutConfig

# File name (within ``rlgammon/agents/saved_agents``) of the calibrated TD value network: its
# win / gammon / backgammon probability components are grounded, so it is the right model for both
# strong play and (crucially) the doubling-cube layer.
CALIBRATED_MODEL = "td-calibrated-077c912f-18c5-4c02-98a7-8f64254922be-(1500).pt"
# Default expectiminimax search depth in decision plies (2-ply = look-ahead over the opponent's reply).
DEFAULT_MAX_DEPTH = 2
# Default truncation depth (decision plies) of the optional rollout leaf evaluator: short, since each
# rollout already sits at a search leaf and is re-run for every frontier node.
DEFAULT_ROLLOUT_MAX_DEPTH = 5
# Default number of independent playouts per rollout-backed leaf evaluation.
DEFAULT_ROLLOUT_TRIALS = 80
# Default seed for the rollout evaluator's (reproducible) dice stream.
DEFAULT_ROLLOUT_SEED = 0


@dataclass(frozen=True)
class StrongAgentConfig:
    """
    Immutable configuration bundle for :func:`build_strong_agent`.

    :param max_depth: expectiminimax search depth in decision plies (``>= 1``; 2 = 2-ply look-ahead)
    :param use_star2: whether to enable star2 chance-node pruning in the expectiminimax search
    :param use_rollouts: wrap the composite leaf evaluator in a truncated-rollout evaluator (stronger,
        much slower) instead of using the static net at search leaves
    :param rollout_trials: number of playouts per rollout when ``use_rollouts`` is set
    :param rollout_max_depth: truncation depth (decision plies) per rollout when ``use_rollouts`` is set
    :param rollout_seed: seed for the rollout evaluator's reproducible dice stream
    :param use_cube: build the agent with the doubling-cube decision methods enabled
    :param match_ctx: the match context the cube decisions are evaluated under (defaults to money play)
    """

    max_depth: int = DEFAULT_MAX_DEPTH
    use_star2: bool = True
    use_rollouts: bool = False
    rollout_trials: int = DEFAULT_ROLLOUT_TRIALS
    rollout_max_depth: int = DEFAULT_ROLLOUT_MAX_DEPTH
    rollout_seed: int = DEFAULT_ROLLOUT_SEED
    use_cube: bool = False
    match_ctx: MatchContext | None = None


class StrongAgent(PlanningAgent):
    """A search-driven planning agent that optionally also makes doubling-cube decisions.

    The move policy is inherited unchanged from :class:`~rlgammon.agents.planning_agent.PlanningAgent`
    (every move is the root action of a fresh search). When built with a cube delegate this class
    additionally answers :meth:`should_double` / :meth:`should_take` / :meth:`cube_action` by forwarding
    to a :class:`~rlgammon.agents.td_agent.TDAgent` built on the *same* calibrated network, so cube and
    checker play share one evaluation.
    """

    def __init__(self, planner: BaseSearch, color: int = WHITE, *,
                 cube_delegate: TDAgent | None = None,
                 match_ctx: MatchContext | None = None) -> None:
        """
        Construct the strong agent around a search planner and an optional cube delegate.

        :param planner: the search algorithm used to choose checker moves
        :param color: 0 or 1 representing which player the agent controls (WHITE=0, BLACK=1)
        :param cube_delegate: a TD agent on the same net used for cube decisions, or ``None`` to
            disable the cube methods (which then raise :class:`CubeDisabledError`)
        :param match_ctx: the match context cube decisions are evaluated under (``None`` = money play)
        """
        super().__init__(planner, color=color)
        self._cube_delegate = cube_delegate
        self._match_ctx = match_ctx if match_ctx is not None else MatchContext()

    def should_double(self, state: GameState, cube: CubeState | None = None) -> bool:
        """
        Return whether the on-roll agent should offer a double in ``state``.

        :param state: the decision-node state with the agent on roll
        :param cube: the current cube state from the agent's perspective (``None`` = centred 1-cube)
        :return: whether the agent should double, per its calibrated net and configured match context
        :raises CubeDisabledError: if the agent was built without the cube layer enabled
        """
        return self.cube_action(state, cube) in (CubeAction.DOUBLE_TAKE, CubeAction.DOUBLE_PASS)

    def should_take(self, state: GameState, cube: CubeState | None = None) -> bool:
        """
        Return whether the agent (the taker) should take an offered double in ``state``.

        :param state: the decision-node state with the agent (the taker) on roll
        :param cube: the pre-double cube state from the agent's perspective (``None`` = centred 1-cube)
        :return: whether the agent should take rather than pass
        :raises CubeDisabledError: if the agent was built without the cube layer enabled
        """
        delegate = self._require_cube()
        return delegate.should_take(state, cube if cube is not None else CubeState(), self._match_ctx)

    def cube_action(self, state: GameState, cube: CubeState | None = None) -> CubeAction:
        """
        Return the agent's full cube decision (no-double / too-good / double-take / double-pass).

        :param state: the decision-node state with the agent on roll
        :param cube: the current cube state from the agent's perspective (``None`` = centred 1-cube)
        :return: the doubler's cube action under the configured match context
        :raises CubeDisabledError: if the agent was built without the cube layer enabled
        """
        delegate = self._require_cube()
        return delegate.cube_action(state, cube if cube is not None else CubeState(), self._match_ctx)

    def _require_cube(self) -> TDAgent:
        """
        Return the cube delegate, raising if the cube layer was not enabled.

        :return: the TD agent used for cube decisions
        :raises CubeDisabledError: if no cube delegate was supplied at construction
        """
        if self._cube_delegate is None:
            raise CubeDisabledError
        return self._cube_delegate


class CubeDisabledError(Exception):
    """Raised when a cube decision is requested from a strong agent built without the cube layer."""

    def __init__(self) -> None:
        """Construct the error with a default message pointing at the ``use_cube`` option."""
        super().__init__("Cube decisions are disabled; build the strong agent with use_cube=True.")


def _leaf_evaluator(net_evaluator: ValueNetEvaluator, config: StrongAgentConfig) -> Evaluator:
    """
    Build the search's leaf evaluator: the phase-aware composite, optionally wrapped in rollouts.

    The base leaf evaluator always routes endgames to the exact bear-off specialist and CONTACT
    positions to the value net. When ``config.use_rollouts`` is set, that composite is further wrapped
    in a truncated-rollout evaluator (stronger but much slower) whose own truncation leaves bootstrap
    the composite.

    :param net_evaluator: the value-network leaf evaluator for CONTACT positions
    :param config: the strong-agent configuration
    :return: the leaf evaluator the search will score frontier states with
    """
    composite = CompositeEvaluator(net_evaluator)
    if not config.use_rollouts:
        return composite
    rollout_config = RolloutConfig(num_trials=config.rollout_trials, max_depth=config.rollout_max_depth,
                                   seed=config.rollout_seed, variance_reduction=True)
    return RolloutEvaluator(composite, rollout_config,
                            rng_factory=lambda: np.random.default_rng(config.rollout_seed))


def build_strong_agent(pre_made_model_file_name: str = CALIBRATED_MODEL,
                       config: StrongAgentConfig | None = None, color: int = WHITE) -> StrongAgent:
    """
    Build the strongest near-term agent: a calibrated value net under a phase-aware expectiminimax.

    The agent scores disengaged leaves with the exact bear-off specialist and contact leaves with the
    calibrated net (optionally a truncated rollout of it), and chooses moves by a star-minimax
    expectiminimax search of the configured depth. With ``config.use_cube`` it also answers cube
    decisions through a TD agent on the same net.

    :param pre_made_model_file_name: file name (within ``rlgammon/agents/saved_agents``) of the
        calibrated TD value network to load (defaults to the shipped calibrated checkpoint)
    :param config: the strong-agent configuration (defaults to 2-ply, star2 on, no rollouts, no cube)
    :param color: 0 or 1 representing which player the agent controls (WHITE=0, BLACK=1)
    :return: a ready-to-play :class:`StrongAgent`
    :raises SearchDepthError: if ``config.max_depth`` is less than 1
    """
    settings = config if config is not None else StrongAgentConfig()
    td_agent = TDAgent(pre_made_model_file_name=pre_made_model_file_name, color=color)
    net = td_agent.get_model()
    evaluator = _leaf_evaluator(ValueNetEvaluator(net), settings)
    planner = StarMinimax(evaluator, settings.max_depth, use_star2=settings.use_star2)
    cube_delegate = td_agent if settings.use_cube else None
    return StrongAgent(planner, color=color, cube_delegate=cube_delegate, match_ctx=settings.match_ctx)
