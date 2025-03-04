"""Implement a logger class for storing data about the training process."""

import logging
from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt

from rlgammon.trainer.logger.logger_types import LoggerData

logging.basicConfig(level=logging.INFO)


class Logger:
    """Create a logger for storing data about the training process."""

    def __init__(self) -> None:
        """Construct a logger by composing it with the python logging library, and initializing an empty data list."""
        self.logger = logging.getLogger("rl-training-logger")
        self.num_items, self.load_episode, self.load_step = 0, 0, 0
        self.info: LoggerData = {"episodes": [0], "steps": [0], "win_rate": [0.0]}

    def update_log(self, episode: int, steps: int, win_rate: float) -> None:
        """
        Add data to the log .

        :param episode: current episode of the training process
        :param steps: current steps of the training process
        :param win_rate: current win rate achieved during tests
        """
        self.num_items += 1
        self.info["episodes"].append(episode)
        self.info["steps"].append(steps)
        self.info["win_rate"].append(win_rate)

    def print_log(self) -> None:
        """Print the most recent logger data to the termial using the python logging library."""
        break_line = "=" * 10 + "\n"
        performance_msg = ("\nBeen training for:\n"
                   f"Episodes: {self.info['episodes'][-1]}\n"
                   f"Steps: {self.info['steps'][-1]}\n"
                   "Current performance:\n"
                   f"Win rate: {self.convert_win_rate_to_percent(self.info['win_rate'][-1])}\n")
        logging_message = performance_msg + break_line
        self.logger.info(logging_message)

    @staticmethod
    def convert_win_rate_to_percent(win_rate:float) -> str:
        """
        Utility function to convert fractional win rate to percent of one data point.

        :param win_rate: fractional win rate to be converted
        :return: the provided win rate represented as a string precentage
        """
        return str(round(win_rate * 100, 2)) + "%"

    def convert_win_rate_log_to_percent(self) -> list[str]:
        """
        Utility function to convert fractional win rate to percent of a list of data point.

        :return: list of string win rates represented as percentages
        """
        return [f"{round(win_rate * 100, 2)}%" for win_rate in self.info["win_rate"]]

    def graph_log(self, logging_choice: str = "episode") -> None:
        """
        Graph all the data in the logger.

        :param logging_choice: determine the x-axis of the graph
        """
        # Set up plt figure parameters
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_ylabel("win rate (%)", fontsize=16)

        if logging_choice == "episode":
            ax.set_title("Win rate through episodes", fontsize=20)
            ax.plot(self.info["episodes"], self.convert_win_rate_log_to_percent(),
                    linewidth=5, color="green")
            ax.set_xlabel("episodes", fontsize=16)
            plt.show()
        elif logging_choice == "step":
            ax.set_title("Win rate through steps", fontsize=20)
            ax.plot(self.info["steps"], self.convert_win_rate_log_to_percent(),
                    linewidth=5, color="green")
            ax.set_xlabel("steps", fontsize=16)
            plt.show()
        else:
            msg = "Invalid logging choice"
            raise ValueError(msg)

    def save(self) -> None:
        """Save the logger to a file, with the current time as differentiating name."""
        logger_name = f"logger-{time.time()}.pkl"
        logger_file_path = Path(__file__).parent
        logger_file_path = logger_file_path.joinpath("saved_loggers/")
        path = logger_file_path.joinpath(logger_name)
        with path.open("wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        """Clear all data from the logger."""
        self.info = {"episodes": [0], "steps": [0], "win_rate": [0.0]}
        self.num_items, self.load_episode, self.load_step = 0, 0, 0
