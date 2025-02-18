import json

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid


# TODO Add Agent to training loop


class StepTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()

    def load_parameters(self, json_parameters: str) -> None:
        parameters = json.loads(json_parameters)
        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            raise ValueError("Invalid parameters")

    def train(self, agent: BaseAgent) -> None:
        if not self.is_ready_for_training():
            raise NoParametersError


        env = BackgammonEnv()
        for episode in range(self.parameters["episodes"]):
            env.reset()
            done = False
            trunc = False
            i = 0
            while not done and not trunc:
                dice = env.roll_dice()
                actions = agent.choose_move(env, dice)

