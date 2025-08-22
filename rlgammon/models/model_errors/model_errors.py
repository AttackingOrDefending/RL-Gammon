"""File implementing errors associated with models."""

class NoLayersErrorError(Exception):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__("A model can't be initialized without providing any layers. "
                         "Please provide at least one layer and an activation function for each layer.")


class InvalidNumberOfActivationFunctionsError(Exception):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__("Each model layer needs an activation function. There must be one (and only one) "
                         "activation function per model layer.")

class ModelNotProvidedToEvaluatorError(Exception):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__("A model must be provided to the evaluator before using it.")