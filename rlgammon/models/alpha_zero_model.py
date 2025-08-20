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
        # reset the gradients
        self.zero_grad()
        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with th.no_grad():
            actor_loss = th.nn.CrossEntropyLoss(action_probs_batch, actor_probs)
            critic_loss = th.nn.MSELoss(reward_batch, critic_value)
            loss = actor_loss + critic_loss
            loss.backward()
            self.optimizer.step()

        if (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
        self.lr_step_current_counter += 1