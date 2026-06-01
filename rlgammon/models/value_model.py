"""File implementing the TD-Gammon value network and its TD(lambda) weight update."""
import numpy as np
import torch as th
from torch import nn

from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_errors.model_errors import (
    EligibilityTracesNotInitializedError,
    ValueHeadConfigError,
)
from rlgammon.models.model_types import ValueHead
from rlgammon.rlgammon_types import Features

# Size of the board-feature input vector fed to the value network.
N_INPUT_FEATURES = 198
# Number of cumulative win/loss probability components produced by the equity head.
N_EQUITY_COMPONENTS = 5
# WHITE-centric terminal returns separating single / gammon / backgammon outcomes (full scoring).
SINGLE_POINTS = 1
GAMMON_POINTS = 2
BACKGAMMON_POINTS = 3


class TDGammonNet(BaseModel):
    """Value network mapping board features to a scalar equity for the perspective player."""

    def __init__(self, lr: float = 0.1, lamda: float = 0.7, hidden: int = 128,
                 value_head: ValueHead = ValueHead.EQUITY_SIGMOID, seed: int = 123, dtype: str = "float32") -> None:
        """
        Construct the value network for the requested output head and initialize TD(lambda) state.

        :param lr: learning rate
        :param lamda: trace decay parameter (how much to value distant states)
        :param hidden: number of units in the hidden layer
        :param value_head: which output head to build (equity-sigmoid or scalar-tanh)
        :param seed: seed for the torch and python random number generators
        :param dtype: the data type of the model
        :raises ValueHeadConfigError: if the requested value head is not supported
        """
        layer_list: list[nn.Module]
        activation_list: list[nn.ReLU | nn.Sigmoid | nn.Tanh | nn.Softmax]
        match value_head:
            case ValueHead.EQUITY_SIGMOID:
                layer_list = [nn.Linear(N_INPUT_FEATURES, hidden), nn.Linear(hidden, N_EQUITY_COMPONENTS)]
                activation_list = [nn.Sigmoid(), nn.Sigmoid()]
            case ValueHead.SCALAR_TANH:
                layer_list = [nn.Linear(N_INPUT_FEATURES, hidden), nn.Linear(hidden, 1)]
                activation_list = [nn.Sigmoid(), nn.Tanh()]
            case _:
                raise ValueHeadConfigError

        super().__init__(lr, layer_list, activation_list, seed, dtype)
        self.lamda = lamda
        self.value_head = value_head
        self.initialized = False
        self.eligibility_traces: list[th.Tensor] | None = None
        self._outcome_initialized = False
        self.outcome_traces: list[th.Tensor] | None = None
        self.device = "cpu"

    def to_device(self, device: str) -> "TDGammonNet":
        """
        Move the network and re-create the eligibility traces on the requested torch device.

        This is an optional accelerator for the multi-output update; the default ``"cpu"`` path is
        unchanged. The (small) per-output traces are rebuilt on the device so the in-place update in
        :meth:`update_outcome_weights` stays device-consistent.

        :param device: the torch device to move the network onto (``"cpu"`` or ``"cuda"``)
        :return: this network, moved onto ``device`` (for call chaining)
        """
        self.device = device
        self.to(th.device(device))
        if self._outcome_initialized:
            self.init_outcome_traces()
        return self

    def raw_outputs(self, x: Features) -> th.Tensor:
        """
        Run a forward pass and return the raw head outputs before equity combination.

        The forward pass builds its input on :attr:`device`; for the default ``"cpu"`` device this is
        identical to the base behaviour, while ``"cuda"`` keeps the input on the GPU with the weights.

        :param x: board-feature input to the model
        :return: the 5-vector in (0, 1) for the equity head, or the 1-vector in (-1, 1) for the scalar head
        """
        if self.device == "cpu":
            return super().forward(x)
        tensor_x = th.as_tensor(np.array(x, dtype=self.np_type), device=th.device(self.device))
        for i, layer in enumerate(self.linears):
            tensor_x = layer(tensor_x)
            if i < self.num_activations:
                tensor_x = self.activation_list[i](tensor_x)
        for i in range(self.num_layers, self.num_activations):
            tensor_x = self.activation_list[i](tensor_x)
        return tensor_x

    @staticmethod
    def combine_equity(raw: th.Tensor) -> th.Tensor:
        """
        Combine the 5 cumulative probability outputs into a scalar equity in (-3, 3).

        The components are cumulative: o0=P(win any), o1=P(win>=gammon), o2=P(win backgammon),
        o3=P(lose>=gammon), o4=P(lose backgammon). This yields +1 for a single win, +2 for a
        gammon and +3 for a backgammon (and the corresponding negatives for losses).

        :param raw: the raw 5-vector head output (batched ``(N, 5)`` or unbatched ``(5,)``)
        :return: the scalar equity (batched ``(N,)`` or an unbatched scalar tensor)
        """
        return (2 * raw[..., 0] - 1) + raw[..., 1] + raw[..., 2] - raw[..., 3] - raw[..., 4]

    @staticmethod
    def outcome_target(returns_white: float) -> th.Tensor:
        """
        Build the cumulative win/loss 5-vector target from WHITE's signed terminal return.

        The components are cumulative masses (see :meth:`combine_equity`): a positive return fills the
        win components up to its magnitude, a negative return fills the loss components. This is the
        TD(lambda) terminal target that grounds every probability component individually, so that the
        property ``combine_equity(outcome_target(r)) == r`` holds for ``r`` in ``+-1, +-2, +-3``.

        :param returns_white: WHITE's signed terminal return (``+-1`` win, ``+-2`` gammon, ``+-3`` backgammon)
        :return: the detached cumulative 5-vector ``(o0, o1, o2, o3, o4)`` from WHITE's perspective
        """
        target = [0.0, 0.0, 0.0, 0.0, 0.0]
        points = round(returns_white)
        if points >= SINGLE_POINTS:
            # WHITE wins: o0=P(win any); o1, o2 stack the gammon and backgammon masses.
            target[0] = 1.0
            target[1] = 1.0 if points >= GAMMON_POINTS else 0.0
            target[2] = 1.0 if points >= BACKGAMMON_POINTS else 0.0
        elif points <= -SINGLE_POINTS:
            # WHITE loses: o0 stays 0 (no win); o3, o4 stack the gammon and backgammon loss masses.
            target[3] = 1.0 if points <= -GAMMON_POINTS else 0.0
            target[4] = 1.0 if points <= -BACKGAMMON_POINTS else 0.0
        return th.tensor(target, dtype=th.get_default_dtype())

    def forward(self, x: Features) -> th.Tensor:
        """
        Make a forward pass and return the scalar value for the perspective player.

        :param x: board-feature input to the model
        :return: the scalar equity in (-3, 3) for the equity head, or the scalar in (-1, 1) for the scalar head
        """
        raw = self.raw_outputs(x)
        if self.value_head == ValueHead.EQUITY_SIGMOID:
            return self.combine_equity(raw)
        return raw.squeeze(-1)

    def init_eligibility_traces(self) -> None:
        """Initialize the eligibility traces to per-parameter zero tensors."""
        trace_dtype = th.float64 if self.np_type == np.float64 else th.float32
        self.eligibility_traces = [th.zeros(weights.shape, dtype=trace_dtype, requires_grad=False)
                                    for weights in self.parameters()]
        self.initialized = True

    def update_weights(self, p: th.Tensor, p_next: th.Tensor | float) -> float:
        """
        Update weights with the undiscounted (gamma=1.0) TD(lambda) algorithm.

        :param p: model evaluation for the current state (carries the gradient)
        :param p_next: model evaluation for the next state, or the final reward if terminal
        :return: the scalar TD error of the update
        :raises EligibilityTracesNotInitializedError: if the eligibility traces were not initialized
        """
        if not self.initialized or self.eligibility_traces is None:
            raise EligibilityTracesNotInitializedError

        # Reset the gradients and compute the derivative of p w.r.t. the parameters.
        self.zero_grad()
        p.backward()  # type: ignore[no-untyped-call]

        with th.no_grad():
            # Only p carries the gradient; p_next is a fixed bootstrap target, so detach/scalarize it.
            td_error = float(p_next) - float(p)
            for i, weights in enumerate(self.parameters()):
                if weights.grad is None:
                    continue
                # z <- lambda * z + (grad of p w.r.t. w); gamma is fixed at 1.0 (undiscounted).
                self.eligibility_traces[i] = self.lamda * self.eligibility_traces[i] + weights.grad
                # w <- w + alpha * td_error * z
                weights.add_(self.lr * td_error * self.eligibility_traces[i])

        if self.lr_scheduler is not None and (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
        self.lr_step_current_counter += 1
        return float(td_error)

    def init_outcome_traces(self) -> None:
        """
        Initialize the per-output eligibility traces to zero tensors.

        For every parameter a single trace tensor of shape ``(N_EQUITY_COMPONENTS, *param.shape)`` is
        created, so ``outcome_traces[i][k]`` is the eligibility trace of output ``k`` with respect to
        parameter ``i``. These back the multi-output TD(lambda) update in
        :meth:`update_outcome_weights`, leaving the scalar :attr:`eligibility_traces` untouched.
        """
        trace_dtype = th.float64 if self.np_type == np.float64 else th.float32
        self.outcome_traces = [
            th.zeros((N_EQUITY_COMPONENTS, *weights.shape), dtype=trace_dtype,
                     device=weights.device, requires_grad=False)
            for weights in self.parameters()
        ]
        self._outcome_initialized = True

    def update_outcome_weights(self, prediction: th.Tensor, target: th.Tensor) -> float:
        """
        Update weights with an undiscounted multi-output TD(lambda) algorithm on the 5-vector head.

        Each of the five cumulative probability outputs is trained as its own value function: per
        output ``k`` the gradient of ``prediction[k]`` w.r.t. every parameter is accumulated into its
        own eligibility trace, then every parameter is moved by ``alpha * sum_k delta[k] * trace[k]``
        where ``delta = target - prediction``. This grounds each probability component individually
        (calibration), unlike :meth:`update_weights` which only supervises the scalar equity.

        :param prediction: the raw 5-vector head output for the current state (carries the gradient)
        :param target: the detached 5-vector bootstrap/terminal target (the next afterstate or outcome)
        :return: the summed squared TD error of the update (``sum_k delta[k]**2``)
        :raises EligibilityTracesNotInitializedError: if the outcome traces were not initialized
        """
        if not self._outcome_initialized or self.outcome_traces is None:
            raise EligibilityTracesNotInitializedError

        parameters = list(self.parameters())
        for k in range(N_EQUITY_COMPONENTS):
            # Reset gradients and get d(prediction[k])/d(w); keep the graph for the remaining outputs.
            self.zero_grad()
            prediction[k].backward(retain_graph=k < N_EQUITY_COMPONENTS - 1)  # type: ignore[no-untyped-call]
            with th.no_grad():
                for i, weights in enumerate(parameters):
                    if weights.grad is None:
                        continue
                    # z_k <- lambda * z_k + (grad of output k w.r.t. w); gamma is fixed at 1.0.
                    self.outcome_traces[i][k] = self.lamda * self.outcome_traces[i][k] + weights.grad

        with th.no_grad():
            # Only prediction carries the gradient; the target is a fixed bootstrap, so detach it.
            delta = (target - prediction).detach()
            for i, weights in enumerate(parameters):
                # w <- w + alpha * sum_k delta[k] * z_k (broadcast delta over each output's trace).
                update = (delta.view(N_EQUITY_COMPONENTS, *([1] * weights.dim())) * self.outcome_traces[i]).sum(dim=0)
                weights.add_(self.lr * update)

        if self.lr_scheduler is not None and (self.lr_step_current_counter + 1) % self.lr_step_count == 0:
            self.lr_scheduler.step()
        self.lr_step_current_counter += 1
        return float((delta ** 2).sum())
