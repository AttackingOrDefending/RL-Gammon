from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError


class StepTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()

    def train(self) -> None:
        if not self.is_ready_for_training():
            raise NoParametersError

        for episode in range(self.parameters["episodes"]):
            pass
