"""A DQN agent for backgammon."""

from functools import cache

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart
from rlgammon.buffers import UniformBuffer

from torch import nn
import torch


class DQN(nn.Module):
    """A simple DQN value network for backgammon."""

    def __init__(self):
        """Initialize the DQN value network."""
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(52, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """Forward pass through the DQN value network."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(BaseAgent):
    """A DQN agent for backgammon."""
    def __init__(self, filename: str = None):
        """Initialize the DQN agent."""
        super(DQNAgent, self).__init__()
        self.value_network = DQN()
        if filename:
            self.value_network.load_state_dict(torch.load(filename))

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

        best_moves = max(scores_per_move, key=lambda x: x[0])[1]
        return best_moves

    def train(self, replay_buffer: UniformBuffer):
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

    def clear_cache(self):
        """Clear the cache of the evaluate_position method."""
        self.evaluate_position.cache_clear()

    @cache
    def evaluate_position(self, board: BackgammonEnv) -> float:
        """Evaluate a position using the DQN value network."""
        state = torch.tensor(board.get_input(), dtype=torch.int8)
        return self.value_network(state)
