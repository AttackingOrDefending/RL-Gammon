"""File implementing errors associated with models."""

class EligibilityTracesNotInitializedError(Exception):
    """
    Class implementing the error caused by attempted training of
    a td-agent without initializing eligibility traces.
    """

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A td-agent can't be trained before initializing the eligibility traces."
                         "Call 'agent.episode_setup()' to properly prepare the agent for training.")


class ValueHeadConfigError(Exception):
    """Class implementing the error caused by configuring a value model with an unknown value head."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("The provided value head is not a supported ValueHead. "
                         "Use ValueHead.EQUITY_SIGMOID or ValueHead.SCALAR_TANH.")
