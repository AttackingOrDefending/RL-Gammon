"""TODO."""
from rlgammon.models.base_model import BaseModel
from rlgammon.models.model_types import ActivationList, LayerList


def model_factory(layer_list: LayerList, activation_list: ActivationList) -> BaseModel:
    """
    TODO.

    :return:
    """
    # Convert None list to empty lists to simplify logic
    layer_list = layer_list if layer_list else []
    activation_list = activation_list if activation_list else []
    return  None
