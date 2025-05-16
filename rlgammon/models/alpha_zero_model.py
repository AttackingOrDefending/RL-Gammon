"""TODO."""
import numpy as np
from numpy.typing import NDArray

from rlgammon.models.mcts_model import MCTSModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Feature


class AlphaZeroModel(MCTSModel):
    """TODO."""

    def __init__(self, lr: float, layer_list: LayerList, activation_list: ActivationList,
                 seed: int=123, dtype: str = "float32") -> None:
        """TODO."""
        super().__init__(lr, layer_list, activation_list, seed, dtype)

    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """TODO."""
        value, policy = self.forward(state)
        value = value.detach().numpy()
        policy = policy.detach().numpy()

        policy[mask == 0] = 0
        policy /= np.sum(policy)
        return value, policy
