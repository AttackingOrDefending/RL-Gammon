import json
from abc import abstractmethod
from typing import Any

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid


class BaseTrainer:
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}

    @abstractmethod
    def train(self) -> None:
        """
        TODO
        """
        raise NotImplementedError

    def load_parameters(self, json_parameters: str) -> None:
        parameters = json.loads(json_parameters)
        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            raise ValueError("Invalid parameters")

    def is_ready_for_training(self) -> bool:
        if self.parameters == {}:
            return False
        return True
