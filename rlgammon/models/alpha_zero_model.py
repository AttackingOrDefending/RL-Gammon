"""TODO."""
import numpy as np
from numpy.typing import NDArray
import torch as th

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
        value_np = value.detach().numpy()
        policy_np = policy.detach().numpy()

        policy_np[mask == 0] = 0
        policy_np /= np.sum(policy)
        return value_np, policy_np

    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """TODO."""
        pass
