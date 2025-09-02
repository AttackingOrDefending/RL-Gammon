"""File implementing errors associated with models."""

class NoLayersErrorError(Exception):
    """Class implementing the error occurring when attempting to create a model with no layers."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A model can't be initialized without providing any layers. "
                         "Please provide at least one layer and an activation function for each layer.")


class InvalidNumberOfActivationFunctionsError(Exception):
    """
    Class implementing the error occurring when attempting to create a model
    with an invalid number of activation functions.
    """

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("Each model layer needs an activation function. There must be one (and only one) "
                         "activation function per model layer.")

class ModelNotProvidedToEvaluatorError(Exception):
    """Class implementing the error occurring when attempting to run the evaluator without first providing a model."""

    def __init__(self) -> None:
        """Construct the error with a default message."""
        super().__init__("A model must be provided to the evaluator before using it.")
