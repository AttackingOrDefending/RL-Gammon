"""TODO."""

import logging
from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt

from rlgammon.trainer.logger.logger_types import LoggerData


class Logger:
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        self.logger = logging.getLogger("rl-training-logger")
        self.num_items, self.load_episode, self.load_step = 0, 0, 0
        self.info: LoggerData = {"episodes": [0], "steps": [0], "win_rate": [0.0]}

    def update_log(self, episode: int, steps: int, win_rate: float) -> None:
        """TODO."""
        self.num_items += 1
        self.info["episodes"].append(episode)
        self.info["steps"].append(steps)
        self.info["win_rate"].append(win_rate)

    def print_log(self) -> None:
        """TODO.""" # TEST
        break_line = "\n=" * 10
        performance_msg = ("Been training for:\n"
                   f"Episodes: {self.info['episodes'][-1]}"
                   f"Steps: {self.info['steps'][-1]}"
                   "Current performance:\n"
                   f"Win rate: {self.info['win_rate'][-1]}")
        logging_message = break_line + performance_msg + break_line
        self.logger.info(logging_message)

    def graph_log(self) -> None:
        """TODO."""
        raise NotImplementedError

    def save(self) -> None:
        """TODO.""" # TEST
        print(Path(__file__))
        buffer_name = f"logger-{time.time()}.pkl"
        buffer_file_path = Path(__file__).parent
        buffer_file_path = buffer_file_path.joinpath("saved_loggers/")
        path = buffer_file_path.joinpath(buffer_name)
        with path.open("wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        raise NotImplementedError

    def clear(self) -> None:
        """TODO."""
        self.info = {"episodes": [0], "steps": [0], "win_rate": [0.0]}
        self.num_items, self.load_episode, self.load_step = 0, 0, 0
