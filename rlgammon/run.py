"""Run the trainer."""

from rlgammon.agents.double_dqn_agent import DoubleDQNAgent
from rlgammon.trainer.step_trainer import StepTrainer

if __name__ == "__main__":
    agent = DoubleDQNAgent()
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
