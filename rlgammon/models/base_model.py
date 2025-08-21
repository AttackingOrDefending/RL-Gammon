"""Base class for all models used in agents."""
from abc import abstractmethod
import random

import numpy as np
import torch as th
from torch import nn

from rlgammon.models.model_types import ActivationList, BaseOutput, LayerList
from rlgammon.models.raw_model import RawModel
from rlgammon.rlgammon_types import Feature

# TODO FIND A WAY TO INCORPORATE VALUE AND POLICY HEAD INTO THIS MODEL

class BaseModel(nn.Module):
    """Class defining the interface of all models or implementing their common functionalities."""

    def __init__(self, lr: float, layer_list: LayerList, activation_list: ActivationList) -> None:
        """
        Construct a base torch model with the provided set of layers and activation functions, and
        parameters.
        Note: layers and activations are interleaved 1 by 1, with the remaining activations filled at the end.

        :param lr: learning rate of a model
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        """
        super().__init__()

        # Set the layers and activation functions of the model
        self.model = RawModel(layer_list, activation_list)

        self.lr = lr
        self.lr_step_count = 100
        self.lr_step_current_counter = 0
        self.decay_rate = 0.96

        self.optimizer = th.optim.Adam(params=list(self.parameters()), lr=self.lr)
        self.lr_scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.decay_rate)

    def forward(self, x: Feature | th.Tensor) -> BaseOutput:
        """
        Make a forward pass through the model with the given data as input.

        :param x: input to the model
        :return: model output
        """
        x = th.FloatTensor(x)
        return self.model(x)  # type: ignore[no-any-return]

    @abstractmethod
    def update_weights(self, p: th.Tensor, p_next: th.Tensor | int) -> float:
        """
        Update the weights of the model from the provided state-values.

        :param p: model evaluation for the current state
        :param p_next: model evaluation for the next state or if terminal state, the final reward
        :return: loss encountered in the update
        """
        raise NotImplementedError
