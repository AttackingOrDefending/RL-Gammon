class NoParametersError(Exception):
    def __init__(self) -> None:
        super().__init__("Can't train without parameters. Load parameters before attempting to train!")
