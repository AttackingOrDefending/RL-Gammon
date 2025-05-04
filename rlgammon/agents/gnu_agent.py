"""TODO."""
from abc import abstractmethod

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment.gnubg.gnubg_backgammon import GnubgInterface


class GNUAgent(BaseAgent):
    """TODO."""

    @abstractmethod
    def handle_opponent_move(self, gnubg: GnubgInterface) -> None:
        """
        TODO.

        :param gnubg:
        :return:
        """
