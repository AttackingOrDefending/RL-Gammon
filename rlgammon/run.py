"""Run the trainer."""
import random
import sys

import numpy as np
import torch as th

from rlgammon.agents.td_agent import TDAgent  # type: ignore[attr-defined]
from rlgammon.trainer.step_trainer import StepTrainer

# TODO Adjust MCTS float data types based on the accuracy used in the model (float32 / float64)

ARGUMENT_COUNT = 2

if __name__ == "__main__":
    # Configuration
    if len(sys.argv) != ARGUMENT_COUNT + 1:
        msg = ("Invalid number of arguments provided. Please provide an integer for the seed. "
               "Please choose one of the following for dtype: [float32, float64].\n"
               f"The number of provided arguments was: {len(sys.argv) - 1} but was supposed to be 2")
        raise ValueError(msg)

    seed = sys.argv[1]
    dtype = sys.argv[2]

    if seed.isnumeric():
        int_seed = int(seed)
        np.random.seed(int_seed)
        th.manual_seed(int_seed)
        random.seed(int_seed)
    else:
        msg = f"Provided seed {seed} is not a number"
        raise TypeError(msg)

    match dtype:
        case "float32":
            th.set_default_dtype(th.float32)
        case "float64":
            th.set_default_dtype(th.float64)
        case _:
            msg = (f"Provided dtype {dtype} is not supported. "
                   f"Please choose one of the following for dtype: [float32, float64].\n")
            raise ValueError(msg)

    agent = TDAgent(
        layer_list=[
            th.nn.Linear(198, 128),
            th.nn.Linear(128, 6),
        ],
        activation_list=[
            th.nn.ReLU(),
            th.nn.Softmax(dim=-1),
        ],
    )
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    trainer.train(agent)
