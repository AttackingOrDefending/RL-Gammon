"""A buffer with uniform sampling."""

from pathlib import Path
import pickle
from typing import Any
from uuid import UUID

import numpy as np

from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.buffers.buffer_types import BufferBatch
from rlgammon.rlgammon_types import Input, MovePart


class UniformBuffer(BaseBuffer):
    """Class implementing a buffer with uniform sampling."""

    def __init__(self, observation_shape: tuple[int, ...], action_shape: int, capacity: int) -> None:
        """
        Constructor for the UniformBuffer, that initializes a base buffer for storing observations.

        :param observation_shape: the shape of the environment states.
        :param action_shape: the shape of the environment actions.
        :param capacity: the number of samples that can maximally be stored in the buffer
        """
        super().__init__(observation_shape, action_shape, capacity)

    def record(self, state: Input, next_state: Input, action: MovePart,
               reward: float, done: bool, player: int, player_after: int, action_info: Any) -> None:  # noqa: ANN401
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
        current_index = self.update_counter % self.capacity
        self.state_buffer[current_index] = state
        self.new_state_buffer[current_index] = next_state
        self.action_buffer[current_index] = action
        self.reward_buffer[current_index] = reward
        self.done_buffer[current_index] = done
        self.player_buffer[current_index] = player
        self.player_after_buffer[current_index] = player_after
        self.action_info_buffer[current_index] = action_info

        self.update_counter += 1

    def has_element_count(self, element_count: int) -> bool:
        """
        Method to check if the buffer contains at least the specified amount of elements.
        By comparing the update counter with the argument.

        :param element_count: element count to check
        :return: boolean, indicating if the buffer has at least the specified element count
        """
        return self.update_counter >= element_count

    def get_batch(self, batch_size: int) -> BufferBatch:
        """
        Get a batch of data from the buffer making a map with random indexes for the numpy arrays.

        :param batch_size: the number of samples to return
        :return: a dict with the following keys: "state", "next_state", "action", "reward", "done",
        "player", "player_after" each having as their value a numpy array with batch size amount of elements
        """
        index_map = np.random.choice(np.arange(min(self.update_counter, self.capacity)), size=batch_size)
        batch_state = self.state_buffer[index_map]
        batch_next_state = self.new_state_buffer[index_map]
        batch_action = self.action_buffer[index_map]
        batch_reward = self.reward_buffer[index_map]
        batch_done = self.done_buffer[index_map]
        batch_player = self.player_buffer[index_map]
        batch_player_after = self.player_after_buffer[index_map]
        batch_action_info = [self.action_info_buffer[i] for i in index_map]

        return {
            "state": batch_state,
            "next_state": batch_next_state,
            "action": batch_action,
            "reward": batch_reward,
            "done": batch_done,
            "player": batch_player,
            "player_after": batch_player_after,
            "action_info": batch_action_info
        }

    def clear(self) -> None:
        """Clear the contents of the buffer by filling all numpy arrays with zeros and resetting the update counter."""
        self.state_buffer.fill(0)
        self.new_state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)
        self.action_info_buffer = [None] * self.capacity

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
        self.action_info_buffer = buffer.action_info_buffer

    def save(self, training_session_id: UUID, session_save_count: int) -> None:
        """
        Save the buffer to a file, with the current time as differentiating name.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        """
        buffer_name = f"uniform-buffer-{training_session_id}-({session_save_count}).pkl"
        buffer_file_path = Path(__file__).parent
        buffer_file_path = buffer_file_path.joinpath("saved_buffers/")
        buffer_file_path.mkdir(parents=True, exist_ok=True)
        path = buffer_file_path.joinpath(buffer_name)
        buffer_file_path.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
