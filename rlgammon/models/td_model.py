"""TODO."""
import torch as th

from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_errors.model_errors import EligibilityTracesNotInitializedError


class TDModel(BaseModel):
    """TODO."""

    def __init__(self, lr: float, lamda: float, seed: int=123) -> None:
        """TODO."""
        super().__init__(lr, lamda, seed)
        self.initialized = False
        self.eligibility_traces = None

    def init_eligibility_traces(self) -> None:
        """TODO."""
        self.eligibility_traces = [th.zeros(weights.shape, requires_grad=False) for weights in list(self.parameters())]
        self.initialized = True

    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        TODO.

        :param p:
        :param p_next:
        :return:
        """
        if not self.initialized:
            raise EligibilityTracesNotInitializedError

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
        return td_error
