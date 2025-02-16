"""Base class for all buffers used in RL-training."""

from abc import abstractmethod

from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.rlgammon_types import Input, MovePart


class BaseBuffer:
    """Base class for all buffers used for training."""

    @abstractmethod
    def record(self, state: Input, next_state: Input, action: MovePart, reward: int, done: bool) -> None:
        """
        Store the environment observation into the buffer.

        :param state: environment state at the recorded step
        :param next_state: the environment state after performing the action at the recorded step
        :param action: the action performed at the recorded step
        :param reward: the reward obtained at the recorded step
        :param done: boolean indicating if the episode ended at the recorded step
        """
        raise NotImplementedError


    @abstractmethod
    def get_batch(self, batch_size: int) -> BufferBatch:
        """
        Get a batch of data from the buffer.

        :param batch_size: the number of samples to return
        :return: a dict with the following keys: "state", "next_state", "action", "reward", "done",
        each of which having as their value a numpy array with batch size amount of elements
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear the contents of the buffer."""
        raise  NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the buffer from the given path.

        :param path: filepath to the file where the buffer is stored
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """Save the buffer to a file."""
        raise NotImplementedError
