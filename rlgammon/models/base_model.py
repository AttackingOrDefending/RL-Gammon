"""TODO."""
from abc import abstractmethod
import random
from uuid import UUID

import numpy as np
import torch as th
from torch import nn

from rlgammon.models.model_types import ActivationList, LayerList


class BaseModel(nn.Module):
    """TODO."""

    def __init__(self, lr: float, seed: int=123,
                 layer_list: LayerList = None, activation_list: ActivationList = None) -> None:
        """
        TODO.

        :param lr:
        :param lamda:
        :param seed:
        :param layer_list:
        :param activation_list:
        """
        super().__init__()
        self.lr = lr
        self.start_episode = 0

        self.layer_list = layer_list
        self.activation_list = activation_list

        self.fc1 = nn.Linear(198, 128)
        self.fc3 = nn.Linear(128, 1)

        self.eligibility_traces = None
        self.optimizer = None

        th.manual_seed(seed)
        random.seed(seed)

    def forward(self, x):
        x = th.from_numpy(np.array(x, dtype=np.float32))
        x = th.relu(self.fc1(x))
        x = th.sigmoid(self.fc3(x))
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
        TODO>.

        :param p:
        :param p_next:
        :return:
        """
        raise NotImplementedError
