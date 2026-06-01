"""File with implementations of errors which could occur while using the planning package."""


class SearchDepthError(Exception):
    """Class implementing the error occurring when a search is created with an invalid maximum depth."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The maximum search depth must be at least 1. "
                         "Provide a `max_depth` >= 1 when constructing a search.")


class NoLegalActionsError(Exception):
    """Class implementing the error occurring when a decision node has no legal actions."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A decision node was reached with no legal actions, yet it is neither "
                         "terminal nor a chance node. Check the game state before searching.")
