"""."""

import pytest

from rlgammon.buffers import UniformBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


@pytest.fixture
def buffer_init() -> UniformBuffer:
    """."""
    capacity = 10_000
    env = BackgammonEnv()
    buffer = UniformBuffer(env.observation_shape, env.action_shape, capacity)

    reward1 = 0
    state1 = env.get_input()
    env.flip()
    next_state1 = env.get_input()

    action1: MovePart = (2, 2) # TODO FIX
    done1 = False
    buffer.record(state1, next_state1, action1, reward1, done1)
    return buffer

@pytest.mark.parametrize("buffer_init", [])
def test_record(buffer: UniformBuffer) -> None:
    """."""
    assert 1 == 1

def test_has_element_count(buffer: UniformBuffer) -> None:
    """."""
    assert buffer.has_element_count(1)
    assert buffer.has_element_count(2)

def test_has_not_element_count(buffer: UniformBuffer) -> None:
    """."""
    assert not buffer.has_element_count(64)

def test_get_batch(buffer: UniformBuffer) -> None:
    """."""
    batch = buffer.get_batch(2)
    # TODO: CHECK IF DATA THE SAME
    assert 1 == 1
