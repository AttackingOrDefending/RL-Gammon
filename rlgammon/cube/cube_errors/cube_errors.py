"""File with implementations of errors which could occur while using the doubling-cube layer."""


class CubelessModelError(Exception):
    """Class implementing the error occurring when a non-equity-sigmoid model is used for cube logic."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The doubling-cube layer requires a value network with the EQUITY_SIGMOID head "
                         "(it decomposes the cumulative 5-vector into win/gammon/backgammon masses). "
                         "Build the model with ValueHead.EQUITY_SIGMOID.")


class InvalidProbabilityVectorError(Exception):
    """Class implementing the error occurring when a probability 5-vector is malformed."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A cube probability vector must hold exactly 5 cumulative sigmoids "
                         "(P(win), P(win>=gammon), P(win bg), P(lose>=gammon), P(lose bg)).")
