"""File implementing errors associated with agents."""

class AlphaZeroNotSetupError(Exception):
    """
    Class implementing the error caused by attempted use of an alpha-zero-agent
    without initializing the MCTS tree.
    """

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("An alpha-zero-agent can't be used before initializing the MCTS tree."
                         "Call agent.episode_setup() to properly prepare the agent for training.")


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
