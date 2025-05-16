"""Base class for all models used in agents."""
from abc import abstractmethod
import random

import numpy as np
import torch as th
from torch import nn

from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import Features


class BaseModel(nn.Module):
    """Class defining the interface of all models or implementing their common functionalities."""

    def __init__(self, lr: float, layer_list: LayerList, activation_list: ActivationList,
                 seed: int=123, dtype: str = "float32") -> None:
        """
        Construct a base torch model with the provided set of layers and activation functions, and
        parameters.
        Note: layers and activations are interleaved 1 by 1, with the remaining activations filled at the end.

        :param lr: learning rate of a model
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        :param seed: seed for random number generator of torch and the python random package
        :param dtype: the data type of the model
        """
        super().__init__()

        # Set the layers and activation functions of the model
        self.activation_list = activation_list
        self.linears = nn.ModuleList(layer_list)
        self.num_layers = len(layer_list) if layer_list else 0
        self.num_activations = len(activation_list) if activation_list else 0

        self.lr = lr
        self.lr_step_count = 100
        self.lr_step_current_counter = 0
        self.decay_rate = 0.96

        self.optimizer = th.optim.Adam(params=list(self.parameters()), lr=self.lr) if self.num_layers != 0 else None
        self.lr_scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=self.decay_rate) if self.num_layers != 0 else None  # type: ignore[arg-type]

        # Set the data type of the models
        self.np_type = np.float32
        th.set_default_dtype(th.float32)
        self.float()
        if dtype == "float64":
            self.np_type = np.float64  # type: ignore[assignment]
            th.set_default_dtype(th.float64)
            self.double()

        # Set seed of the random number generators
        th.manual_seed(seed)
        random.seed(seed)

    def forward(self, x: Features) -> th.Tensor:
        """
        Make a forward pass through the model with the given data as input.

        :param x: input to the model
        :return: model output
        """
        x = th.from_numpy(np.array(x, dtype=self.np_type))  # type: ignore[assignment]
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < self.num_activations:
                x = self.activation_list[i](x)
        for i in range(self.num_layers, self.num_activations):
            x = self.activation_list[i](x)
        return x  # type: ignore[return-value]

    @abstractmethod
    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        Update the weights of the model from the provided state-values.

        :param p: model evaluation for the current state
        :param p_next: model evaluation for the next state or if terminal state, the final reward
        :return: loss encountered in the update
        """
        raise NotImplementedError
