"""File with implementations of errors which could occur while using the endgame package."""

from rlgammon.endgame.endgame_types import CHECKERS_PER_SIDE, HOME_BOARD_SIZE


class InvalidHomeConfigError(Exception):
    """Class implementing the error occurring when a one-sided home-board configuration is malformed."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__(
            f"A home-board configuration must hold exactly {HOME_BOARD_SIZE} non-negative checker "
            f"counts summing to at most {CHECKERS_PER_SIDE} (the rest are borne off). "
            "Build it with `home_config_from_points`.",
        )


class ObservationTensorLengthError(Exception):
    """Class implementing the error occurring when an observation tensor has an unexpected length."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__(
            "The observation tensor does not have the expected backgammon length (200). "
            "Decode only real OpenSpiel backgammon observation tensors with this package.",
        )


class NonBearoffPositionError(Exception):
    """Class implementing the error occurring when an exact two-sided bear-off is requested off-database."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__(
            "The exact two-sided bear-off computation requires both sides to have every checker in "
            "their home board (and none on the bar). Check the phase before calling it.",
        )
