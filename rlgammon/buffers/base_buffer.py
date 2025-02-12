from abc import abstractmethod

from rlgammon.rlgammon_types import MovePart, Input
from rlgammon.buffers.buffer_types import BufferBatch


class BaseBuffer:
    @abstractmethod
    def record(self, state: Input, next_state: Input, action: MovePart, reward: int, done: bool) -> None:
        """
        TODO

        :param state:
        :param next_state:
        :param action:
        :param reward:
        :param done:
        :return:
        """

        raise NotImplementedError


    @abstractmethod
    def get_batch(self, batch_size: int) -> BufferBatch:
        """
        TODO

        :param batch_size:
        :return:
        """

        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        TODO
        """

        raise  NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        TODO

        :param path:
        """

        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """
        TODO
        """

        raise NotImplementedError
