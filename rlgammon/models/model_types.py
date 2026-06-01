"""File storing types associated with models."""
from enum import Enum
from typing import TypedDict

import torch as th

LayerList = list[th.nn.Module]
ActivationList = list[th.nn.ReLU | th.nn.Sigmoid | th.nn.Tanh | th.nn.Softmax]


class ValueHead(Enum):
    """Enumeration of the supported value-network output heads."""

    EQUITY_SIGMOID = "EQ"
    SCALAR_TANH = "TANH"

    @staticmethod
    def get_enum_from_string(string_to_convert: str) -> "ValueHead":
        """
        Convert a string, found e.g. in JSON parameters, to a ValueHead enum.

        :param string_to_convert: the string value to convert
        :return: the corresponding enum, if none found, return null
        """
        match string_to_convert:
            case "EQ":
                return ValueHead.EQUITY_SIGMOID
            case "TANH":
                return ValueHead.SCALAR_TANH
            case _:
                return None  # type: ignore[return-value]


class EquityComponents(TypedDict):
    """Cumulative win/loss probability components produced by the equity head."""

    p_win: float
    p_win_gammon: float
    p_win_backgammon: float
    p_lose_gammon: float
    p_lose_backgammon: float
