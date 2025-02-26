import json

from rlgammon.agents.dqn_agent import DQNAgent
from rlgammon.trainer.step_trainer import StepTrainer


if __name__ == '__main__':
    agent = DQNAgent()
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
