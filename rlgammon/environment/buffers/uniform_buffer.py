import numpy as np

from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.rlgammon_types import MovePart, Input



class UniformBuffer(BaseBuffer):
    """
    TODO
    """

    def __init__(self, capacity: int) -> None:
        """
        TODO

        :param capacity:
        """

        self.capacity = capacity
        self.update_counter = 0

        self.state_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.new_state_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.action_buffer = np.zeros(shape=(self.capacity,), dtype=np.int8)
        self.reward_buffer = np.zeros(shape=self.capacity, dtype=np.int8)
        self.done_buffer = np.zeros(shape=self.capacity, dtype=np.bool)

    def update(self, state: Input, next_state: Input, action: MovePart, reward: int, done: bool) -> None:
        """
        TODO

        :param state:
        :param next_state:
        :param action:
        :param reward:
        :param done:
        :return:
        """

        current_index = self.update_counter % self.capacity
        self.state_buffer[current_index] = state
        self.new_state_buffer[current_index] = next_state
        self.action_buffer[current_index] = action
        self.reward_buffer[current_index] = reward
        self.done_buffer[current_index] = done

    def get_batch(self, batch_size: int) -> BufferBatch:
        """
        TODO

        :param batch_size:
        :return:
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
        TODO
        """

        self.state_buffer.fill(0)
        self.new_state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)
