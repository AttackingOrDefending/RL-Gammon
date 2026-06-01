"""Build a :class:`PlanningAgent` from a value model and a :class:`SearchConfig`.

This lives apart from ``search_factory`` (which the ``planning`` package eagerly re-exports) so that the
``planning`` -> ``agents`` dependency it introduces does not create an import cycle; it is the seam used
to "think deeper at test time" by handing a trained value network a heavier search than training used.
"""

from rlgammon.agents.planning_agent import PlanningAgent
from rlgammon.models.base_model import BaseModel
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator
from rlgammon.planning.planning_types import PossibleSearch, SearchConfig
from rlgammon.planning.search_factory import WrongSearchTypeError, create_search
from rlgammon.rlgammon_types import WHITE


def build_planning_agent(model: BaseModel, config: SearchConfig, color: int = WHITE) -> PlanningAgent:
    """
    Build a planning agent that scores leaves with ``model`` and searches per ``config``.

    The search-type-specific knobs are forwarded as keyword arguments: ``use_star2`` for star-minimax,
    and ``num_simulations``/``c_uct`` for MCTS.

    :param model: the trained value network used to evaluate non-terminal frontier states
    :param config: the search configuration (algorithm, depth and per-algorithm hyper-parameters)
    :param color: 0 or 1 representing which player the agent controls
    :return: a planning agent wrapping a freshly constructed search of the configured type
    :raises SearchDepthError: if ``config.max_depth`` is less than 1
    :raises WrongSearchTypeError: if ``config.search_type`` is not a known PossibleSearch
    """
    evaluator = ValueNetEvaluator(model)
    match config.search_type:
        case PossibleSearch.STAR_MINIMAX:
            kwargs: dict[str, object] = {"use_star2": config.use_star2}
        case PossibleSearch.MCTS:
            kwargs = {"num_simulations": config.num_simulations, "c_uct": config.c_uct}
        case _:
            raise WrongSearchTypeError
    planner = create_search(config.search_type, evaluator, config.max_depth, **kwargs)
    return PlanningAgent(planner, color=color)
