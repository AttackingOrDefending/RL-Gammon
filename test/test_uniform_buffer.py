"""Testing of the Uniform Buffer."""
import numpy as np
import pytest

from rlgammon.buffers import UniformBuffer
from rlgammon.environment import BackgammonEnv


@pytest.fixture
def buffer() -> UniformBuffer:
    """Initialize buffer with pre-added data."""
    capacity = 10_000
    env = BackgammonEnv()
    buffer = UniformBuffer(env.observation_shape, env.action_shape, capacity)

    reward1 = 0
    state1 = env.get_input()
    action1 = (6, (12, 18))
    env.flip()
    next_state1 = env.get_input()
    done1 = False
    buffer.record(state1, next_state1, action1[1], reward1, done1, 1, -1)

    reward2 = 0
    state2 = env.get_input()
    action2 = (1, (12, 13))
    env.flip()
    next_state2 = env.get_input()
    done2 = True
    buffer.record(state2, next_state2, action2[1], reward2, done2, -1, 1)
    return buffer


def test_record(buffer: UniformBuffer) -> None:
    """
    Test adding an observation tuple to the buffer.

    :param buffer: Pre-initialized buffer.
    """
    buffer_size = buffer.update_counter
    env = BackgammonEnv()

    reward = 0
    state = env.get_input()
    action = (2, (2, 4))
    env.flip()
    next_state = env.get_input()
    done = False
    buffer.record(state, next_state, action[1], reward, done, 1, -1)

    assert buffer_size + 1 == buffer.update_counter


def test_has_element_count(buffer: UniformBuffer) -> None:
    """
    Test positively the has_element_count method. Should return true iff buffer-elements >= n.

    :param buffer: Pre-initialized buffer.
    """
    assert buffer.has_element_count(1)
    assert buffer.has_element_count(2)


def test_has_not_element_count(buffer: UniformBuffer) -> None:
    """
    Test negatively the has_element_count method. Should return false iff buffer-elements < n.

    :param buffer: Pre-initialized buffer.
    """
    assert not buffer.has_element_count(64)


def test_contains_state(buffer: UniformBuffer) -> None:
    """
    Test positively whether the buffer contains a state.

    :param buffer: Pre-initialized buffer.

    """
    assert buffer.contains_state(buffer.state_buffer[0])


def test_not_contains_state(buffer: UniformBuffer) -> None:
    """
    Test negatively whether the buffer contains a state.

    :param buffer: Pre-initialized buffer.

    """
    state = np.ones(shape=buffer.state_buffer[0].shape) * -789
    assert not buffer.contains_state(state)


def test_contains_new_state(buffer: UniformBuffer) -> None:
    """
    Test positively whether the buffer contains a new-state.

    :param buffer: Pre-initialized buffer.

    """
    assert buffer.contains_state(buffer.new_state_buffer[0])


def test_not_contains_new_state(buffer: UniformBuffer) -> None:
    """
    Test negatively whether the buffer contains a new-state.

    :param buffer: Pre-initialized buffer.

    """
    new_state = np.ones(shape=buffer.new_state_buffer[0].shape) * -789
    assert not buffer.contains_state(new_state)


def test_get_batch(buffer: UniformBuffer) -> None:
    """
    Test whether the data received from buffer is valid.

    :param buffer: Pre-initialized buffer.
    """
    batch = buffer.get_batch(1)
    state = batch["state"][0]
    next_state = batch["next_state"][0]
    assert buffer.contains_state(state)
    assert buffer.contains_next_state(next_state)
