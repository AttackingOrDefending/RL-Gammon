#type: ignore  # noqa: PGH003

"""File implementing a model used in td training."""
import torch as th

from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Feature


class TDModel(BaseModel):
    """Class implementing a TD model used in td training."""

    def __init__(self, lr: float, gamma: float, lamda: float, layer_list: LayerList, activation_list: ActivationList,
                 seed: int=123, dtype: str = "float32") -> None:
        """
        Construct a td model by first constructing a base torch model,
        and then initializing td specific parameters.
        Note: layers and activations are interleaved 1 by 1, with the remaining activations filled at the end.

        :param lr: learning rate
        :param gamma: future reward discount
        :param lamda: trace decay parameters (how much to value distant states)
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        :param seed: seed for random number generator of torch and the python random package
        :param dtype: the data type of the model
        """
        super().__init__(lr, layer_list, activation_list, seed, dtype)
        self.gamma = gamma
        self.lamda = lamda
        self.eligibility_traces = None

    def forward(self, x: Feature) -> th.Tensor:
        """
        Forward pass of the model.

        :param x: input to the model
        :return: output of the model
        """
        x = super().forward(x)
        return x[0] * -3 + x[1] * -2 + x[2] * -1 + x[3] * 1 + x[4] * 2 + x[5] * 3

    def init_eligibility_traces(self) -> None:
        """Initialize the eligibility traces."""
        self.eligibility_traces = [th.zeros(weights.shape, dtype=th.float64, requires_grad=False)
                                    for weights in list(self.parameters())]

    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        Update weights according to the td-lambda algorithm.

        :param p: model evaluation for the current state
        :param p_next: model evaluation for the next state or if terminal state, the final reward
        :return: loss encountered in the update
        """
        # reset the gradients
        self.zero_grad()
        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with th.no_grad():
            td_error = p_next - p

            # get the parameters of the model
            parameters = list(self.parameters())

            for i, weights in enumerate(parameters):
                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.eligibility_traces[i] = self.gamma * self.lamda * self.eligibility_traces[i] + weights.grad
                # w <- w + alpha * td_error * z
                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.copy_(new_weights)

        if (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_lr()[0]
        self.lr_step_current_counter += 1
        return td_error
