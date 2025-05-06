"""Run the trainer."""
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.step_trainer import StepTrainer

if __name__ == "__main__":
    agent = TDAgent(layer_list=[th.nn.Linear(198, 128),
                                th.nn.Linear(128, 128),
                                th.nn.Linear(128, 6),
                                ],
                    activation_list=[th.nn.ReLU,
                                     th.nn.ReLU,
                                     th.nn.Tanh,
                                     ],
                    )
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
