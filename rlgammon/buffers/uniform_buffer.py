"""A buffer with uniform sampling."""

from pathlib import Path
import pickle
import time

import numpy as np

from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.rlgammon_types import Input, MoveList


class UniformBuffer(BaseBuffer):
    """Class implementing a buffer with uniform sampling."""

    def __init__(self, observation_shape: tuple[int, ...], max_action_shape: int, capacity: int = 10_000) -> None:
        """
        Constructor for the UniformBuffer, that initializes the counter, and all the numpy arrays for storing data.

        :param capacity: the number of samples that can maximally be stored in the buffer
        """
        self.capacity = capacity
        self.update_counter = 0
        self.max_action_shape = max_action_shape

        self.state_buffer = np.zeros(shape=(self.capacity, *observation_shape), dtype=np.int8)
        self.new_state_buffer = np.zeros(shape=(self.capacity, *observation_shape), dtype=np.int8)
        self.action_buffer = np.zeros(shape=(self.capacity, max_action_shape), dtype=np.int8)
        self.reward_buffer = np.zeros(shape=self.capacity, dtype=np.int8)
        self.done_buffer = np.zeros(shape=self.capacity, dtype=np.bool)

    def record(self, state: Input, next_state: Input, action: MoveList, reward: int, done: bool) -> None:
        """
        Store the environment observation into the buffer.

        :param state: environment state at the recorded step
        :param next_state: the environment state after performing the action at the recorded step
        :param action: the action performed at the recorded step
        :param reward: the reward obtained at the recorded step
        :param done: boolean indicating if the episode ended at the recorded step
        """
        current_index = self.update_counter % self.capacity
        self.state_buffer[current_index] = state
        self.new_state_buffer[current_index] = next_state
        numpy_action = np.ones(self.max_action_shape, dtype=np.int8) * -2  # -1 is used for bear off
        for i, (roll, move) in enumerate(action):
            numpy_action[i * 2] = move[0]
            numpy_action[i * 2 + 1] = move[1]
        self.action_buffer[current_index] = numpy_action
        self.reward_buffer[current_index] = reward
        self.done_buffer[current_index] = done

        self.update_counter += 1

    def get_batch(self, batch_size: int) -> BufferBatch:
        """
        Get a batch of data from the buffer making a map with random indexes for the numpy arrays.

        :param batch_size: the number of samples to return
        :return: a dict with the following keys: "state", "next_state", "action", "reward", "done",
        each of which having as their value a numpy array with batch size amount of elements
        """
        index_map = np.random.choice(np.arange(min(self.update_counter, self.capacity)), size=batch_size)
        batch_state = self.state_buffer[index_map]
        batch_next_state = self.new_state_buffer[index_map]
        batch_action = self.action_buffer[index_map]
        batch_reward = self.reward_buffer[index_map]
        batch_done = self.done_buffer[index_map]

        return {
            "state": batch_state,
            "next_state": batch_next_state,
            "action": batch_action,
            "reward": batch_reward,
            "done": batch_done,
        }

    def clear(self) -> None:
        """Clear the contents of the buffer by filling all numpy arrays with zeros and resetting the update counter."""
        self.state_buffer.fill(0)
        self.new_state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)

        self.update_counter = 0

    def load(self, buffer_name: str) -> None:
        """
        Load the buffer with the given name.

        :param buffer_name: name of the saved buffer to load
        """
        buffer_file_path = "../rlgammon/buffers/saved_buffers/"
        path = Path(buffer_file_path + buffer_name)

        with path.open("rb") as f:
            buffer = pickle.load(f)

        self.update_counter = buffer.update_counter
        self.state_buffer = buffer.state_buffer
        self.new_state_buffer = buffer.new_state_buffer
        self.action_buffer = buffer.action_buffer
        self.reward_buffer = buffer.reward_buffer
        self.done_buffer = buffer.done_buffer

    def save(self) -> None:
        """Save the buffer to a file, with the current time as differentiating name."""
        buffer_name = f"uniform-buffer-{time.time()}.pkl"
        buffer_file_path = "../rlgammon/buffers/saved_buffers/"
        path = Path(buffer_file_path + buffer_name)
        with path.open("wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
