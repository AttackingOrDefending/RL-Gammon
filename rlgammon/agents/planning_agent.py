"""An agent that chooses moves by delegating to a search/planning algorithm."""

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.base_search import BaseSearch
from rlgammon.rlgammon_types import WHITE, ActionSetGNU


class PlanningAgent(BaseAgent):
    """An agent that plugs a search planner into the agent interface."""

    def __init__(self, planner: BaseSearch, color: int = WHITE) -> None:
        """
        Construct the planning agent around a search planner.

        :param planner: the search algorithm used to choose moves
        :param color: 0 or 1 representing which player the agent controls
        """
        super().__init__(color)
        self._planner = planner

    def episode_setup(self) -> None:
        """A planning agent needs no setup, therefore the function does nothing."""

    def choose_move(self, actions: list[int] | ActionSetGNU,  # noqa: ARG002
                    state: GameState) -> int:
        """
        Choose a move by searching from the current state and returning the best action.

        :param actions: set of all possible actions to choose from (unused; the planner searches)
        :param state: the current game state to choose a move for
        :return: the planner's best action from ``state``
        """
        return self._planner.search(state).best_action
