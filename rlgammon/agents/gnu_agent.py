"""File defining the interface of an agent capable of playing against a gnubg agent."""
from abc import abstractmethod

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment.gnubg.gnubg_backgammon import GnubgInterface


class GNUAgent(BaseAgent):
    """Class defining the interface of an agent capable of playing against a gnubg agent."""

    @abstractmethod
    def handle_opponent_move(self, gnubg: GnubgInterface) -> None:
        """
        TODO.

        :param gnubg: the interface used to communicate with gnubg
        """
