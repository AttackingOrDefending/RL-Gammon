import json

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid
from rlgammon.trainer.base_trainer import BaseTrainer


class StepTrainer(BaseTrainer):
    def __init__(self) -> None:
        self.parameters = None

    def train(self) -> None:
        pass

    def load_parameters(self, json_parameters: str) -> None:
        parameters = json.loads(json_parameters)
        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            raise ValueError("Invalid parameters")
