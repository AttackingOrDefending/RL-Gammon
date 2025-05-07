"""Define different types related to the logger."""

from typing import TypedDict


class LoggerData(TypedDict):
    """Define type defining what data the logger holds."""

    episodes: list[int]
    steps: list[int]
    results: list[dict[str, float]]
    training_time: list[float]
