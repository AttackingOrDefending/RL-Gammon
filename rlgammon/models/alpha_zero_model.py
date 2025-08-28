"""TODO."""
import numpy as np
from numpy.typing import NDArray
import torch as th

from rlgammon.models.mcts_model import MCTSModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Feature

# TODO FIX INPUT TO TRAINING BETWEEN ALPHA ZERO AND TD AGENTS!!!
# TODO FIX LIST NO TENSOR PASSED TO MODELS ... -> CHANGE HINTS !!!

class AlphaZeroModel(MCTSModel):
    """TODO."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """TODO."""
        super().__init__(lr, base_layer_list, base_activation_list, policy_layer_list, policy_activation_list,
                         value_layer_list, value_activation_list)

    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """TODO."""
        value, policy = self.forward(state)
        value_np = value.detach().numpy()
        policy_np = policy.detach().numpy()
        policy_np[mask == 0] = 0  # need to get first element, as 2d array is provided as mask
        policy_np /= np.sum(policy_np)
        return value_np, policy_np

    def update_weights(self, mcts_probs: list[th.Tensor], actor_pred_probs: list[th.Tensor],
                       reward_batch: list[th.Tensor], critic_pred_values: list[th.Tensor]) -> float:
        """TODO."""
        # reset the gradients
        self.zero_grad()

        with th.set_grad_enabled(True):
            mcts_probs = th.tensor(mcts_probs)
            reward_batch = th.tensor(reward_batch, dtype=th.float32)

            actor_loss = th.nn.CrossEntropyLoss()(actor_pred_probs, mcts_probs)
            critic_loss = th.nn.MSELoss()(critic_pred_values, reward_batch)

            loss = actor_loss + critic_loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        if (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
        self.lr_step_current_counter += 1
        return loss.item()
