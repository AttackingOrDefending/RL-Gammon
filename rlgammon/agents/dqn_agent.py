# mypy: disable-error-code="no-any-return, no-untyped-call, union-attr, operator, assignment"
"""A DQN agent for backgammon."""

from functools import cache
import pathlib

import torch
from torch import nn

from rlgammon.agents.base_agent import BaseAgent
from rlgammon.buffers import BaseBuffer
from rlgammon.environment import NO_MOVE_NUMBER, BackgammonEnv
from rlgammon.exploration import BaseExploration
from rlgammon.rlgammon_types import MovePart


class DQN(nn.Module):
    """A simple DQN value network for backgammon."""

    def __init__(self) -> None:
        """Initialize the DQN value network."""
        super().__init__()
        self.fc1 = nn.Linear(52, 128, dtype=torch.float16)
        self.fc2 = nn.Linear(128, 128, dtype=torch.float16)
        self.fc3 = nn.Linear(128, 1, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DQN value network."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    """A DQN agent for backgammon."""

    def __init__(self, exploration: BaseExploration, main_filename: str = "value.pt",
                 target_filename: str = "target_value.pt", optimizer_filename: str = "optimizer.pt", lr: float = 0.001,
                 tau: float = 0.01, batch_size: int = 64, gamma: float = 0.99, max_grad_norm: float = 10) -> None:
        """Initialize the DQN agent."""
        super().__init__()
        self.exploration = exploration
        self.main_filename = main_filename
        self.target_filename = target_filename
        self.optimizer_filename = optimizer_filename
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.value_network = DQN()
        self.target_network = DQN()
        self.target_network.load_state_dict(self.value_network.state_dict())
        if main_filename and pathlib.Path(main_filename).exists():
            self.value_network.load_state_dict(torch.load(main_filename))
        if target_filename and pathlib.Path(target_filename).exists():
            self.target_network.load_state_dict(torch.load(target_filename))
        self.optimizer = torch.optim.RMSprop(self.value_network.parameters(), lr=self.lr)
        if optimizer_filename and pathlib.Path(optimizer_filename).exists():
            self.optimizer.load_state_dict(torch.load(optimizer_filename))

    def choose_move(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """Choose a move according to the DQN value network."""
        board_copy = board.copy()
        dice = dice.copy()
        actions = board_copy.get_all_complete_moves(dice)

        scores_per_move = []

        if actions[0][1][0][0] == -NO_MOVE_NUMBER:
            return []
        for board_after_move, moves in actions:
            value = -self.evaluate_position(board_after_move)  # Minimize opponent's value
            scores_per_move.append((value, moves))

        best_move = max(scores_per_move, key=lambda x: x[0])[1]

        return self.exploration.explore(best_move, [move for _, move in actions])

    def train(self, replay_buffer: BaseBuffer) -> None:
        """Train the DQN value network using the replay buffer. We don't use actions as we are building a value network."""
        batch = replay_buffer.get_batch(self.batch_size)
        states = torch.tensor(batch["state"], dtype=torch.float16)
        next_states = torch.tensor(batch["next_state"], dtype=torch.float16)
        rewards = torch.tensor(batch["reward"], dtype=torch.float16)
        dones = torch.tensor(batch["done"], dtype=torch.bool)

        current_q_values = self.value_network(states).squeeze()

        with torch.no_grad():
            next_q_values = -self.target_network(next_states).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.binary_cross_entropy_with_logits(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.value_network.parameters(), strict=True):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # Copy batch norm layer stats as well as they are not part of the parameters.
        for target_param, param in zip(self.target_network.named_buffers(), self.value_network.named_buffers(), strict=True):
            if "running_mean" in target_param[0]:
                target_param[1].copy_(param[1])
            if "running_var" in target_param[0]:
                target_param[1].copy_(param[1])

        # Clear the cache of the evaluate_position method as we have updated the model.
        self.clear_cache()

    def save(self, main_filename: str | None = None, target_filename: str | None = None,
             optimizer_filename: str | None = None) -> None:
        """Save the DQN agent."""
        if main_filename is None:
            main_filename = self.main_filename
        if target_filename is None:
            target_filename = self.target_filename
        if optimizer_filename is None:
            optimizer_filename = self.optimizer_filename
        torch.save(self.value_network.state_dict(), main_filename)
        torch.save(self.target_network.state_dict(), target_filename)
        torch.save(self.optimizer.state_dict(), optimizer_filename)

    def clear_cache(self) -> None:
        """Clear the cache of the evaluate_position method."""
        self.evaluate_position.cache_clear()

    @cache
    def evaluate_position(self, board: BackgammonEnv) -> float:
        """Evaluate a position using the DQN value network."""
        state = torch.tensor(board.get_input(), dtype=torch.float16)
        return float(self.value_network(state))
