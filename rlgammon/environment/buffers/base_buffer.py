from abc import abstractmethod

from rlgammon.rlgammon_types import MovePart, Input


class BaseBuffer:
    @abstractmethod
    def update(self, state: Input, next_state: Input, action: MovePart, reward: int, done: bool) -> None:
        """"""
        raise NotImplementedError

