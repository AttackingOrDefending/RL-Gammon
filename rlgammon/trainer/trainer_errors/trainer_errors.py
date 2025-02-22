"""TODO"""

class NoParametersError(Exception):
    """TODO"""
    def __init__(self) -> None:
        """TODO"""
        super().__init__("Can't train without parameters. Load parameters before attempting to train!")


class WrongBufferTypeError(Exception):
    """TODO"""
    def __init__(self) -> None:
        """TODO"""
        super().__init__("The buffer type you are trying to use is not available right now! "
                         "Please check 'PossibleBuffers' for available buffers!")


class WrongExplorationTypeError(Exception):
    """TODO"""
    def __init__(self) -> None:
        """TODO"""
        super().__init__("The exploration type you are trying to use is not available right now! "
                         "Please check 'PossibleExploration' for available exploration types!")
