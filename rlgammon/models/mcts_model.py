"""TODO."""
from abc import abstractmethod

from rlgammon.models.base_model import BaseModel


class MCTSModel(BaseModel):
    """TODO."""

    def __init__(self, num_actions) -> None:
        """TODO."""

    @abstractmethod
    def inference(self, state, mask):
        """TODO."""
