# mypy: disable-error-code="no-any-return, no-untyped-call, union-attr, operator"
"""A DQN agent for backgammon."""

from functools import cache
import os

import torch
from torch import nn

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.buffers import UniformBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class DQN(nn.Module):
    """A simple DQN value network for backgammon."""

    def __init__(self) -> None:
        """Initialize the DQN value network."""
        super().__init__()
        self.fc1 = nn.Linear(52, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DQN value network."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    """A DQN agent for backgammon."""
    TAU = 0.01

    def __init__(self, main_filename: str = "value.pt", target_filename: str = "target_value.pt") -> None:
        """Initialize the DQN agent."""
        super().__init__()
        self.value_network = DQN()
        self.target_network = DQN()
        self.target_network.load_state_dict(self.value_network.state_dict())
        if main_filename and os.path.exists(main_filename):
            self.value_network.load_state_dict(torch.load(main_filename))
        if target_filename and os.path.exists(target_filename):
            self.target_network.load_state_dict(torch.load(target_filename))

    def choose_move(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """Choose a move according to the DQN value network."""
        board_copy = board.copy()
        dice = dice.copy()
        actions = board_copy.get_all_complete_moves(dice)

        scores_per_move = []

        if actions[0][1][0][0] == -1:
            return []
        for board_after_move, moves in actions:
            value = self.evaluate_position(board_after_move)
            scores_per_move.append((value, moves))

        return max(scores_per_move, key=lambda x: x[0])[1]

    def train(self, replay_buffer: UniformBuffer) -> None:
        """Train the DQN value network using the replay buffer."""
        batch = replay_buffer.get_batch(32)
        states = torch.tensor(batch["state"], dtype=torch.int8)
        next_states = torch.tensor(batch["next_state"], dtype=torch.int8)
        actions = torch.tensor(batch["action"], dtype=torch.int8)
        rewards = torch.tensor(batch["reward"], dtype=torch.int8)
        dones = torch.tensor(batch["done"], dtype=torch.bool)

        values = self.value_network(states).gather(1, actions)
        next_values = self.value_network(next_states).max(1)[0].detach()
        target_values = rewards + 0.99 * next_values * ~dones
        loss = nn.functional.mse_loss(values, target_values.unsqueeze(1))

        self.value_network.zero_grad()
        loss.backward()
        self.value_network.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def clear_cache(self) -> None:
        """Clear the cache of the evaluate_position method."""
        self.evaluate_position.cache_clear()

    @cache
    def evaluate_position(self, board: BackgammonEnv) -> float:
        """Evaluate a position using the DQN value network."""
        state = torch.tensor(board.get_input(), dtype=torch.int8)
        return float(self.value_network(state))
