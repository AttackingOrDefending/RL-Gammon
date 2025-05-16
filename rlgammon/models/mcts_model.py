"""TODO."""
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_types import ActivationList, LayerList, MCTSOutput
from rlgammon.rlgammon_types import Feature


class MCTSModel(BaseModel):
    """TODO."""

    def __init__(self, lr: float, layer_list: LayerList, activation_list: ActivationList,
                 seed: int=123, dtype: str = "float32") -> None:
        """TODO."""
        super().__init__(lr, layer_list, activation_list, seed, dtype)

    def forward(self, x: Feature) -> MCTSOutput:
        pass

    @abstractmethod
    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """TODO."""
