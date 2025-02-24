from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.trainer.step_trainer import StepTrainer


if __name__ == '__main__':
    agent = TrainableAgent()
    trainer = StepTrainer()
    trainer.train(agent)
