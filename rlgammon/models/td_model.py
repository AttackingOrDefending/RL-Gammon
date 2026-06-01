"""File implementing a backward-compatible TD model used in td training."""
from rlgammon.models.model_types import ActivationList, LayerList, ValueHead
from rlgammon.models.value_model import TDGammonNet


class TDModel(TDGammonNet):
    """Backward-compatible thin wrapper around :class:`TDGammonNet` accepting the legacy signature."""

    def __init__(self, lr: float, gamma: float, lamda: float, layer_list: LayerList, activation_list: ActivationList,
                 seed: int = 123, dtype: str = "float32") -> None:
        """
        Construct a td model, delegating to the corrected, undiscounted :class:`TDGammonNet`.

        Note: ``gamma`` is deprecated and ignored (the update is undiscounted), and the externally
        supplied ``layer_list``/``activation_list`` are ignored in favour of the equity-sigmoid head.

        :param lr: learning rate
        :param gamma: deprecated future-reward discount, ignored (the update is undiscounted)
        :param lamda: trace decay parameter (how much to value distant states)
        :param layer_list: deprecated list of layers, ignored in favour of the equity-sigmoid head
        :param activation_list: deprecated list of activations, ignored in favour of the equity-sigmoid head
        :param seed: seed for the torch and python random number generators
        :param dtype: the data type of the model
        """
        del gamma, layer_list, activation_list
        super().__init__(lr=lr, lamda=lamda, value_head=ValueHead.EQUITY_SIGMOID, seed=seed, dtype=dtype)
