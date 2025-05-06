"""Base class for all models used in agents."""
from abc import abstractmethod
import random

import numpy as np
import torch as th
from torch import nn

from rlgammon.models.model_types import ActivationList, LayerList


class BaseModel(nn.Module):
    """Class defining the interface of all models or implementing their common functionalities."""

    def __init__(self, lr: float, seed: int=123,
                 layer_list: LayerList = None, activation_list: ActivationList = None) -> None:
        """
        Construct a base torch model with the provided set of layers and activation functions, and
        parameters.
        Note: layers and activations are interleaved 1 by 1, with the remaining activations filled at the end.

        :param lr: learning rate of a model
        :param seed: seed for random number generator of torch and the python random package
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        """
        super().__init__()
        self.lr = lr

        self.layer_list = layer_list
        self.activation_list = activation_list

        self.fc1 = nn.Linear(198, 128, dtype=th.float64)
        self.fc3 = nn.Linear(128, 1, dtype=th.float64)

        self.optimizer = None

        th.manual_seed(seed)
        random.seed(seed)

    def forward(self, x):
        x = th.from_numpy(np.array(x, dtype=np.float64))
        x = th.relu(self.fc1(x))
        x = th.tanh(self.fc3(x))
        return x

    """
    def forward(self, x: State) -> th.Tensor:

        num_layers = len(self.layer_list)
        num_activations= len(self.activation_list)

        # apply all layers and the corresponding activation function
        for i in range(num_layers):
            x = self.layer_list[i](x)
            if i < num_activations:
                x = self.activation_list[i](x)

        # apply any remaining activation functions if necessary
        if num_activations - num_layers > 0:
            for i in range(num_layers, num_activations):
                x = self.activation_list[i](x)
        return x
    """

    @abstractmethod
    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        Update the weights of the model from the provided state-values.

        :param p: model evaluation for the current state
        :param p_next: model evaluation for the next state or if terminal state, the final reward
        :return: loss encountered in the update
        """
        raise NotImplementedError
