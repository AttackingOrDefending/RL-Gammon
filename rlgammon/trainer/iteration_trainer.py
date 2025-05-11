"""Construct the trainer by initializing its parameters in the BaseTrainer class."""
from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.trainer.base_trainer import BaseTrainer


class IterationTrainer(BaseTrainer):
    """Implementation of trainer where training is performed after each iteration."""

    def __init__(self) -> None:
        """Construct the trainer by initializing its parameters in the BaseTrainer class."""
        super().__init__()

    def train(self, agent: TrainableAgent) -> None:
        """
        TODO.

        :param agent:
        :return:
        """
