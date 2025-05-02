from enum import Enum
from functools import partial

import torch as th
from torch import nn


class LayerType(Enum):
    """TODO."""

    DENSE = nn.Linear
    CONV = nn.Conv2d

    def __call__(self, x: th.Tensor) -> th.Tensor:
        """
        TODO.

        :param x:
        :return:
        """
        return self.value(x)


class ActivationType(Enum):
    """TODO."""

    LINEAR: partial = partial(th.nn.functional.linear)
    RELU: partial = partial(th.relu)
    SIGMOID: partial = partial(th.sigmoid)
    TANH: partial = partial(th.tanh)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        """
        TODO.

        :param x:
        :return:
        """
        return self.value(x)
