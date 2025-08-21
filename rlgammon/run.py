"""Run the trainer."""
import torch as th

from rlgammon.agents.td_agent import TDAgent  # type: ignore[attr-defined]
from rlgammon.trainer.step_trainer import StepTrainer

# TODO Adjust MCTS float data types based on the accuracy used in the model (float32 / float64)

"""
        # Set the data type of the models
        self.np_type = np.float32
        th.set_default_dtype(th.float32)
        self.float()
        if dtype == "float64":
            self.np_type = np.float64  # type: ignore[assignment]
            th.set_default_dtype(th.float64)
            self.double()

        # Set seed of the random number generators
        th.manual_seed(seed)
        random.seed(seed)

"""

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
