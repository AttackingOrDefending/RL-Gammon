"""File with implementations of different errors which could occur while using the trainer."""

class NoParametersError(Exception):
    """Class implementing the error occurring when attempting to train without valid parameters."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("Can't train without parameters. Load parameters before attempting to train!")


class WrongBufferTypeError(Exception):
    """
    Class implementing the error occurring
    when the wrong buffer type has been provided in the parameters.
    """

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The buffer type you are trying to use is not available right now! "
                         "Please check 'PossibleBuffers' for available buffers!")


class WrongExplorationTypeError(Exception):
    """
    Class implementing the error occurring
    when the wrong exploration type has been provided in the parameters.
    """

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The exploration type you are trying to use is not available right now! "
                         "Please check 'PossibleExploration' for available exploration types!")
