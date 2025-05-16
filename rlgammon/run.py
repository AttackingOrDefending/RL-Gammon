"""Run the trainer."""
import torch as th

from rlgammon.agents.td_agent import TDAgent  # type: ignore[attr-defined]
from rlgammon.trainer.step_trainer import StepTrainer

# TODO Adjust MCTS float data types based on the accuracy used in the model (float32 / float64)

if __name__ == "__main__":
    agent = TDAgent(
        layer_list=[
            th.nn.Linear(198, 128),
            th.nn.Linear(128, 6),
        ],
        activation_list=[
            th.nn.ReLU(),
            th.nn.Softmax(dim=-1),
        ],
        dtype="float32",
    )
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
