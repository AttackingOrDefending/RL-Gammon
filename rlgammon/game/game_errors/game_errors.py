"""File with implementations of errors which could occur while using the game engine boundary."""


class EngineNotAvailableError(Exception):
    """Class implementing the error occurring when a requested engine backend can't be imported."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The requested game engine is not available. OpenSpiel (`pyspiel`) could not be "
                         "imported. Install `open_spiel`, or use PossibleEngine.MOCK for tests.")


class WrongEngineTypeError(Exception):
    """Class implementing the error occurring when an unknown engine backend is requested."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The engine type you are trying to use is not available right now! "
                         "Please check 'PossibleEngine' for available engines!")


class NonChanceNodeError(Exception):
    """Class implementing the error occurring when chance operations are attempted on a non-chance node."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("Chance outcomes were requested on a state that is not a chance node. "
                         "Check `state.is_chance_node()` before sampling dice.")


class TerminalStateError(Exception):
    """Class implementing the error occurring when an action is applied to a terminal state."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("An action can't be applied to a terminal state. "
                         "Check `state.is_terminal()` before applying an action.")
