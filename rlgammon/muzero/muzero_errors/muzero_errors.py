"""File with implementations of errors which could occur while using the MuZero package."""


class CodebookSizeError(Exception):
    """Class implementing the error occurring when the chance codebook size is invalid."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The chance codebook size must be at least 1. "
                         "Provide a `codebook_size` >= 1 in the MuZeroConfig.")


class UnrollLengthError(Exception):
    """Class implementing the error occurring when an unroll is requested with an invalid length."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The number of unroll steps must be at least 1. "
                         "Provide an `unroll_steps` >= 1 in the MuZeroConfig.")
