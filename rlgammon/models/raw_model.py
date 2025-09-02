"""
Raw model consisting of just layers and activation functions, enabling a simple forward pass.
Intended to be used to build more complex models.
"""
import torch as th
from torch import nn

from rlgammon.models.model_errors.model_errors import InvalidNumberOfActivationFunctionsError, NoLayersErrorError
from rlgammon.models.model_types import ActivationList, BaseOutput, LayerList
from rlgammon.rlgammon_types import Feature


class RawModel(nn.Module):
    """Class defining the interface of all models or implementing their common functionalities."""

    def __init__(self, layer_list: LayerList, activation_list: ActivationList) -> None:
        """
        Construct a base torch model with the provided set of layers and activation functions, and
        parameters.
        Note: layers and activations are interleaved 1 by 1, with the remaining activations filled at the end.

        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        """
        super().__init__()

        if layer_list is None or activation_list is None or len(layer_list) == 0:
            raise NoLayersErrorError

        if len(layer_list) != len(activation_list):
            raise InvalidNumberOfActivationFunctionsError

        # Set the layers and activation functions of the model
        self.activation_list = activation_list
        self.linears = nn.ModuleList(layer_list)
        self.num_layers = len(layer_list)
        self.num_activations = len(activation_list)

    def forward(self, x: Feature | th.Tensor) -> BaseOutput:
        """
        Make a forward pass through the model with the given data as input.

        :param x: input to the model
        :return: model output
        """
        x = th.FloatTensor(x)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < self.num_activations:
                x = self.activation_list[i](x)
        for i in range(self.num_layers, self.num_activations):
            x = self.activation_list[i](x)
        return x  # type: ignore[return-value]
