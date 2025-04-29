"""Run the trainer."""
from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.step_trainer import StepTrainer

if __name__ == "__main__":
    agent = TDAgent()
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
