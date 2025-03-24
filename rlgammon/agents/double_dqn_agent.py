# mypy: disable-error-code="no-any-return, no-untyped-call, union-attr, operator, assignment"
"""A DQN agent for backgammon."""

from functools import cache
import pathlib
from uuid import UUID

import torch
from torch import nn

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers import BaseBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import MovePart


class DQN(nn.Module):
    """A simple DQN value network for backgammon."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the DQN value network."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64, dtype=torch.float32)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 1, dtype=torch.float32)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.weight.data /= 1000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DQN value network."""
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)


class DoubleDQNAgent(TrainableAgent):
    """A DQN agent for backgammon."""

    def __init__(self, main_filename: str = "ddqn-value",
                 target_filename: str = "ddqn-target_value", optimizer_filename: str = "ddqn-optimizer", lr: float = 0.001,
                 tau: float = 0.01, batch_size: int = 64, gamma: float = 0.99, max_grad_norm: float = 5) -> None:
        """Initialize the DQN agent."""
        super().__init__()
        self.main_filename = main_filename
        self.target_filename = target_filename
        self.optimizer_filename = optimizer_filename
        self.input_dim = BackgammonEnv().observation_shape[0]
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.value_network = DQN(self.input_dim)
        self.target_network = DQN(self.input_dim)
        self.target_network.load_state_dict(self.value_network.state_dict())
        if main_filename and pathlib.Path(main_filename).exists():
            self.value_network.load_state_dict(torch.load(main_filename))
        if target_filename and pathlib.Path(target_filename).exists():
            self.target_network.load_state_dict(torch.load(target_filename))
        self.optimizer = torch.optim.RMSprop(self.value_network.parameters(), lr=self.lr, momentum=0.9, eps=0.001,
                                             centered=True)
        if optimizer_filename and pathlib.Path(optimizer_filename).exists():
            self.optimizer.load_state_dict(torch.load(optimizer_filename))

    def choose_move(self, board: BackgammonEnv, dice: list[int]) -> list[tuple[int, MovePart]]:
        """Choose a move according to the DQN value network."""
        board_copy = board.copy()
        dice = dice.copy()
        actions = board_copy.get_all_complete_moves(dice)

        scores_per_move = []

        if not actions:
            return []
        for board_after_move, moves in actions:
            value = -self.evaluate_position(board_after_move)  # Minimize opponent's value
            scores_per_move.append((value, moves))

        return max(scores_per_move, key=lambda x: x[0])[1]

    def train(self, replay_buffer: BaseBuffer) -> None:
        """Train the DQN value network using the replay buffer. We don't use actions as we are building a value network."""
        batch = replay_buffer.get_batch(self.batch_size)
        states = torch.tensor(batch["state"], dtype=torch.float32)
        next_states = torch.tensor(batch["next_state"], dtype=torch.float32)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32)
        dones = torch.tensor(batch["done"], dtype=torch.bool)

        current_q_values = self.value_network(states).squeeze()

        with torch.no_grad():
            next_q_values = -self.target_network(next_states).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * ~dones

        loss = nn.functional.mse_loss(current_q_values, target_q_values)

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

    def save(self, training_session_id: UUID, session_save_count: int, main_filename: str | None = None,
             target_filename: str | None = None, optimizer_filename: str | None = None) -> None:
        """
        Save the DQN agent.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        :param main_filename: filename where the main network is to be saved
        :param target_filename: filename where the target network is to be saved
        :param optimizer_filename: filename where the optimizer is to be saved
        """
        if main_filename is None:
            main_filename = self.main_filename
        if target_filename is None:
            target_filename = self.target_filename
        if optimizer_filename is None:
            optimizer_filename = self.optimizer_filename

        agent_main_filename = f"{main_filename}-{training_session_id}-({session_save_count}).pt"
        agent_target_filename = f"{target_filename}-{training_session_id}-({session_save_count}).pt"
        agent_optimizer_filename = f"{optimizer_filename}-{training_session_id}-({session_save_count}).pt"

        agent_file_path = pathlib.Path(__file__).parent
        agent_file_path = agent_file_path.joinpath("saved_agents/")
        agent_file_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.value_network.state_dict(), agent_file_path.joinpath(agent_main_filename))
        torch.save(self.target_network.state_dict(), agent_file_path.joinpath(agent_target_filename))
        torch.save(self.optimizer.state_dict(), agent_file_path.joinpath(agent_optimizer_filename))

    def clear_cache(self) -> None:
        """Clear the cache of the evaluate_position method."""
        self.evaluate_position.cache_clear()

    @cache
    def evaluate_position(self, board: BackgammonEnv) -> float:
        """Evaluate a position using the DQN value network."""
        state = torch.tensor(board.get_input(), dtype=torch.float32)
        return float(self.value_network(state))
