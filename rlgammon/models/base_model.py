"""TODO."""
from abc import abstractmethod
from datetime import datetime
import random

import numpy as np
import torch as th
from torch import nn

from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import State


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

    def checkpoint(self, checkpoint_path: str, step: int, name_experiment: str):
        """
        TODO.

        :param checkpoint_path:
        :param step:
        :param name_experiment:
        :return:
        """
        path = checkpoint_path + "/{}_{}_{}.tar".format(name_experiment, datetime.now().strftime("%Y%m%d_%H%M_%S_%f"), step + 1)
        th.save({"step": step + 1, "model_state_dict": self.state_dict(), "eligibility": self.eligibility_traces if self.eligibility_traces else []}, path)
        print(f"\nCheckpoint saved: {path}")

    def load(self, checkpoint_path: str, optimizer=None, eligibility_traces=None):
        """
        TODO.

        :param checkpoint_path:
        :param optimizer:
        :param eligibility_traces:
        :return:
        """
        checkpoint = th.load(checkpoint_path)
        self.start_episode = checkpoint["step"]

        self.load_state_dict(checkpoint["model_state_dict"])

        if eligibility_traces is not None:
            self.eligibility_traces = checkpoint["eligibility"]

        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
