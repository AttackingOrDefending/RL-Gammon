class NoParametersError(Exception):
    """TODO"""
    def __init__(self) -> None:
        """TODO"""
        super().__init__("Can't train without parameters. Load parameters before attempting to train!")
