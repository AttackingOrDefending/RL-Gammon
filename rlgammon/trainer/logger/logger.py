"""TODO."""

import logging

from rlgammon.trainer.logger.logger_types import LoggerData


class Logger:
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        self.num_items = 0
        self.num_items, self.load_epoch, self.load_episode, self.load_step = 0, 0, 0, 0
        self.info: LoggerData = {"episodes": [0], "steps": [0], "win_rate": [0.0]}

    def update_log(self, episode: int, steps: int, win_rate: float) -> None:
        """TODO."""
        self.num_items += 1
        self.info["episodes"].append(episode)
        self.info["steps"].append(steps)
        self.info["win_rate"].append(win_rate)

    def print_log(self) -> None:
        """TODO."""
        raise NotImplementedError

    def graph_log(self) -> None:
        """TODO."""
        raise NotImplementedError

    def save(self) -> None:
        """TODO."""
        raise NotImplementedError

    def clear(self) -> None:
        """TODO."""
        raise NotImplementedError
