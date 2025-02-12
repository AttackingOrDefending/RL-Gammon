import json
from abc import abstractmethod



class BaseTrainer:
    @abstractmethod
    def train(self) -> None:
        """
        TODO
        """
        raise NotImplementedError

    @abstractmethod
    def load_parameters(self, parameters: str) -> None:
        """
        TODO
        :param parameters: JSON string
        """
        raise NotImplementedError
