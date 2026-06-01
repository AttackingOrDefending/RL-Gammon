"""File with implementations of errors which could occur while using the rollout package."""


class RolloutConfigError(Exception):
    """Class implementing the error occurring when a rollout is configured with invalid parameters."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A rollout needs at least one trial and a positive truncation depth. "
                         "Provide a `num_trials` >= 1 and a `max_depth` >= 1 in the RolloutConfig.")


class ChanceRootError(Exception):
    """Class implementing the error occurring when a chance-node rollout is given no perspective."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A rollout from a chance node has no implicit side to move, so its "
                         "`perspective` must be given explicitly (it defaults to the side to move, "
                         "which is undefined at a chance node). Pass `perspective=WHITE`/`BLACK`.")
