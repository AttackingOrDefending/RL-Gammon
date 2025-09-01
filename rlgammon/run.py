"""Run the trainer."""
import random
import sys

import numpy as np
import pyspiel
import torch as th

from rlgammon.agents.alpha_zero_agent import AlphaZeroAgent  # type: ignore[attr-defined]
from rlgammon.agents.td_agent import TDAgent  # type: ignore[attr-defined]
from rlgammon.evaluators.alpha_zero_evaluator import AlphaZeroEvaluator
from rlgammon.trainer.iteration_trainer import IterationTrainer
from rlgammon.trainer.step_trainer import StepTrainer

ARGUMENT_COUNT = 3

if __name__ == "__main__":
    # Configuration
    if len(sys.argv) != ARGUMENT_COUNT + 1:
        msg = ("Invalid number of arguments provided.\n"
               "Please provide an int for the seed.\n"
               "Please choose one of the following for dtype: ['float32', 'float64'].\n"
               "Please choose one of the following for trainer type: ['step', 'iteration']\n"
               f"The number of provided arguments was: {len(sys.argv) - 1} but was supposed to be {ARGUMENT_COUNT}")
        raise ValueError(msg)

    seed = sys.argv[1]
    dtype = sys.argv[2]
    trainer_type = sys.argv[3]

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
            msg = (f"Provided dtype '{dtype}' is not supported.\n"
                   f"Please choose one of the following for dtype: [float32, float64].")
            raise ValueError(msg)

    match trainer_type:
        case "step":
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
        case "iteration":
            evaluator = AlphaZeroEvaluator()
            agent = AlphaZeroAgent(evaluator, pyspiel.load_game("backgammon(scoring_type=full_scoring)"),
                                   4, 500, 0,
                                   base_layer_list=[
                                       th.nn.Linear(198, 512),
                                       th.nn.Linear(512, 256),
                                   ],
                                   base_activation_list=[
                                       th.nn.ReLU(),
                                       th.nn.ReLU(),
                                   ],
                                   policy_layer_list=[
                                       th.nn.Linear(256, 128),
                                       th.nn.Linear(128, 1352),
                                   ],
                                   policy_activation_list=[
                                       th.nn.ReLU(),
                                       th.nn.Softmax(dim=-1),
                                   ],
                                   value_layer_list=[
                                       th.nn.Linear(256, 128),
                                       th.nn.Linear(128, 1),
                                   ],
                                   value_activation_list=[
                                       th.nn.ReLU(),
                                       th.nn.Tanh(),
                                   ])
            agent.mcts_evaluator.provide_model(agent.model)  # type: ignore[arg-type]

            trainer = IterationTrainer()  # type: ignore[assignment]
            trainer.load_parameters("parameters.json")
            trainer.train(agent)

        case _:
            msg = (f"Provided trainer type '{trainer_type}' is not supported.\n"
                   f"Please choose one of the following for trainer type: ['step', 'iteration']\n")
            raise TypeError(msg)
