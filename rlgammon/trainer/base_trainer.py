from abc import abstractmethod


class BaseTrainer:
    @abstractmethod
    def train(self) -> None:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def load_parameters(self) -> None:
        """
        """
        raise NotImplementedError
