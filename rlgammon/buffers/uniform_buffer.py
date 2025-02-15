import pickle
import time

import numpy as np

from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.rlgammon_types import MovePart, Input



class UniformBuffer(BaseBuffer):
    """
    Class implementing a buffer with uniform sampling.
    """

    def __init__(self, capacity: int) -> None:
        """
        Constructor for the UniformBuffer, initializing the counter, and all the numpy arrays for storing data.

        :param capacity: the number of samples that can maximally be stored in the buffer
        """

        self.capacity = capacity
        self.update_counter = 0

        self.state_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.new_state_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.action_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.reward_buffer = np.zeros(shape=self.capacity, dtype=np.int8)
        self.done_buffer = np.zeros(shape=self.capacity, dtype=np.bool)

    def record(self, state: Input, next_state: Input, action: MovePart, reward: int, done: bool) -> None:
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
        self.action_buffer[current_index] = action
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

        return {"state": batch_state,
                "next_state": batch_next_state,
                "action": batch_action,
                "reward": batch_reward,
                "done": batch_done}

    def clear(self) -> None:
        """
        Clear the contents of the buffer by filling all numpy arrays with zeros and resetting the update counter.
        """

        self.state_buffer.fill(0)
        self.new_state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)

        self.update_counter = 0

    def load(self, path: str) -> None:
        """
        Load the buffer from the given path.

        :param path: filepath to the file where the buffer is stored
        """

        with open(path, "rb") as f:
            buffer = pickle.load(f)

        self.update_counter = buffer.update_counter
        self.state_buffer = buffer.state_buffer
        self.new_state_buffer = buffer.new_state_buffer
        self.action_buffer = buffer.action_buffer
        self.reward_buffer = buffer.reward_buffer
        self.done_buffer = buffer.done_buffer

    def save(self) -> None:
        """
        Save the buffer to a file, with the current time as differentiating name.
        """

        buffer_name = f"uniform-buffer-{str(time.time())}.pkl"
        buffer_file_path = "rlgammon/buffers/saved_buffers/"
        with open(buffer_file_path + buffer_name, "wb") as f:
            f.write(pickle.dumps(self))
