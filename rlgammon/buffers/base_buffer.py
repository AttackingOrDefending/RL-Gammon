"""Base class for all buffers used in RL-training."""

from abc import abstractmethod
from uuid import UUID

from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.rlgammon_types import Input, MoveList


class BaseBuffer:
    """Base class for all buffers used for training."""

    @abstractmethod
    def record(self, state: Input, next_state: Input, action: MoveList, reward: float, done: bool) -> None:
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
    def has_element_count(self, element_count: int) -> bool:
        """
        Method to check if the buffer contains at least the specified amount of elements.

        :param element_count: element count to check
        :return: boolean, indicating if the buffer has at least the specified element count
        """

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
    def save(self, training_session_id: UUID, session_save_count: int) -> None:
        """
        Save the buffer to a file.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        """
        raise NotImplementedError
