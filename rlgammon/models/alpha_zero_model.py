"""File implementing an alpha-zero style model."""
import numpy as np
from numpy.typing import NDArray
import torch as th

from rlgammon.models.mcts_model import MCTSModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Feature


class AlphaZeroModel(MCTSModel):
    """Class implementing an alpha-zero style model."""

    def __init__(self, lr: float, base_layer_list: LayerList, base_activation_list: ActivationList,
                 policy_layer_list: LayerList, policy_activation_list: ActivationList,
                 value_layer_list: LayerList, value_activation_list: ActivationList) -> None:
        """
        Construct an alpha-zero style model by creating an actor-critic model.

        :param lr: learning rate
        :param base_layer_list: list of layers to use in the base (shared) network
        :param base_activation_list: list of activations to use in the base (shared) network
        :param policy_layer_list: list of layers to use in the policy network
        :param policy_activation_list: list of activations to use in the policy network
        :param value_layer_list: list of layers to use in the value network
        :param value_activation_list: list of activations to use in the value network
        """
        super().__init__(lr, base_layer_list, base_activation_list, policy_layer_list, policy_activation_list,
                         value_layer_list, value_activation_list)

    def inference(self, state: Feature, mask: NDArray[np.bool]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        TODO.

        :param state:
        :param mask:
        :return:
        """
        value, policy = self.forward(state)
        value_np = value.detach().numpy()
        policy_np = policy.detach().numpy()
        policy_np[mask == 0] = 0  # need to get first element, as 2d array is provided as mask
        policy_np /= np.sum(policy_np)
        return value_np, policy_np

    def update_weights(self, mcts_probs: th.Tensor, reward: th.Tensor, state: Feature, _: Feature, __: bool) -> float:
        """
        Update model weights using data generated during episode runs.
        Use MCTS policy to train policy network, and the reward to train the critic network.

        :param mcts_probs: the policy returned by MCTS
        :param reward: reward obtained by the agent
        :param state: current state of the game
        :param __: unused parameter, included for compatibility
        :param _: unused parameter, included for compatibility
        :return: combined policy and critic loss for this training step
        """
        critic_pred_values, actor_pred_probs = self.forward(state)

        # reset the gradients
        self.zero_grad()

        with th.set_grad_enabled(True):
            actor_loss = th.nn.CrossEntropyLoss()(actor_pred_probs, mcts_probs)
            critic_loss = th.nn.MSELoss()(critic_pred_values, reward)

            loss = actor_loss + critic_loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        if (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
        self.lr_step_current_counter += 1
        return loss.item()  # type: ignore[no-any-return]
