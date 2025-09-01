"""Base class for all buffers used in RL-training."""

from abc import abstractmethod
from typing import Any
from uuid import UUID

import numpy as np

from rlgammon.buffers.buffer_types import BufferBatch, BufferData
from rlgammon.rlgammon_types import Input, MovePart, ObservationShape


class BaseBuffer:
    """Base class for all buffers used for training."""

    def __init__(self, action_shape: int, capacity: int) -> None:
        """
        Constructor for the BaseBuffer, that initializes the counter, and all the numpy arrays for storing data.

        :param observation_shape: the shape of the environment states.
        :param action_shape: the shape of the environment actions.
        :param capacity: the number of samples that can maximally be stored in the buffer
        """
        self.capacity = capacity
        self.update_counter = 0
        self.action_shape = action_shape

        self.state_buffer = np.zeros(shape=(self.capacity, *ObservationShape), dtype=np.float32)
        self.new_state_buffer = np.zeros(shape=(self.capacity, *ObservationShape), dtype=np.float32)
        self.action_buffer = np.zeros(shape=(self.capacity, action_shape), dtype=np.int32)
        self.reward_buffer = np.zeros(shape=self.capacity, dtype=np.float32)
        self.done_buffer = np.zeros(shape=self.capacity, dtype=np.bool)
        self.player_buffer = np.zeros(capacity, dtype=np.int8)
        self.player_after_buffer = np.zeros(capacity, dtype=np.int8)
        self.action_info_buffer: list[Any] = [None] * capacity # list to store any input

    @abstractmethod
    def record(self, state: Input, next_state: Input, action: MovePart,
               reward: float, done: bool, player: int, player_after: int, action_info: Any) -> None:
        """
        Store the environment observation into the buffer.

        :param state: environment state at the recorded step
        :param next_state: the environment state after performing the action at the recorded step
        :param action: the action performed at the recorded step
        :param reward: the reward obtained at the recorded step
        :param done: boolean indicating if the episode ended at the recorded step
        :param player: the player who made the action
        :param player_after: the player who is to player after the action
        :param action_info: information about the action, e.g. for a0 mcts probabilities from search
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
    def get_all_elements(self) -> BufferData:
        """
        Get all elements stored in the buffer.
        :return: all elements stored in the buffer
        """
        raise NotImplementedError

    @abstractmethod
    def create_dataset(self) :
        """
        Create a PyTorch dataset.
        :return: PyTorch dataset
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

    def get_num_elements(self) -> int:
        """TODO."""
        return min(self.update_counter, self.capacity)

    def contains_state(self, state: Input) -> bool:
        """
        Check if the provided state is stored in the buffer's state-array.

        :param state: state to be searched
        :return: true, if state is in the buffer, else false
        """
        return any(np.array_equal(state, self.state_buffer[i]) for i in range(self.state_buffer.shape[0]))

    def contains_next_state(self, new_state: Input) -> bool:
        """
        Check if the provided state is stored in the buffer's new-state-array.

        :param new_state: state to be searched
        :return: true, if state is in the buffer, else false
        """
        return any(np.array_equal(new_state, self.new_state_buffer[i]) for i in range(self.new_state_buffer.shape[0]))
