"""TODO."""

from typing import TypedDict


class LoggerData(TypedDict):
    """TODO."""

    episodes: list[int]
    steps: list[int]
    win_rate: list[float]
