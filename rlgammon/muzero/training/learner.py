"""The Stochastic MuZero learner: owns the optimizer and runs a single gradient step on a batch."""
import pathlib
from uuid import UUID

import torch as th

from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.replay.replay_buffer import Batch
from rlgammon.muzero.training.losses import muzero_loss

# Maximum global gradient norm; gradients are clipped to this before the optimizer step.
GRAD_CLIP_NORM = 10.0


class MuZeroLearner:
    """Own the network and Adam optimizer and apply one Stochastic MuZero gradient step per batch."""

    def __init__(self, config: MuZeroConfig, network: StochasticMuZeroNetwork) -> None:
        """
        Construct the learner around a configuration and a network with an Adam optimizer.

        :param config: the configuration providing the learning rate, weight decay and loss settings
        :param network: the Stochastic MuZero network whose parameters are optimized
        """
        self.config = config
        self.network = network
        self.optimizer = th.optim.Adam(
            self.network.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

    def train_step(self, batch: Batch) -> dict[str, float]:
        """
        Run one optimization step on a batch and return the scalarized component losses.

        The network is put in training mode, the gradients are zeroed, the K-step unrolled loss is
        computed and back-propagated, the global gradient norm is clipped to :data:`GRAD_CLIP_NORM` and
        the optimizer is stepped.

        :param batch: the stacked batch of unroll windows to train on
        :return: a dict mapping each loss key to its scalar float value
        """
        self.network.train()
        self.optimizer.zero_grad()
        batch = self._batch_to_device(batch)
        losses = muzero_loss(self.config, self.network, batch)
        losses["total"].backward()  # type: ignore[no-untyped-call]
        th.nn.utils.clip_grad_norm_(self.network.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        return {key: float(value.detach()) for key, value in losses.items()}

    def _batch_to_device(self, batch: Batch) -> Batch:
        """
        Move every tensor field of a batch onto the network's device for the forward / backward.

        The replay buffer already stacks batches on the configured device, so this is a no-op in the
        common path; it makes the learner robust to being handed a CPU batch (e.g. from a buffer with
        a different device) and is where moving the heavy training tensors on-device is enforced.

        :param batch: the batch whose tensors to move
        :return: a batch whose tensors all live on ``self.network.device``
        """
        device = self.network.device
        return Batch(
            observation=batch.observation.to(device),
            actions=batch.actions.to(device),
            target_values=batch.target_values.to(device),
            target_rewards=batch.target_rewards.to(device),
            target_policies=batch.target_policies.to(device),
            chance_observations=batch.chance_observations.to(device),
            weights=batch.weights.to(device),
        )

    def save(self, training_session_id: UUID, session_save_count: int,
             main_filename: str = "stochastic-muzero") -> None:
        """
        Save the network state dict under the training package's ``saved_agents`` directory.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions so far
        :param main_filename: base name of the file under which the network is saved
        """
        agent_main_filename = f"{main_filename}-{training_session_id}-({session_save_count}).pt"
        agent_file_path = pathlib.Path(__file__).parent.joinpath("saved_agents/")
        agent_file_path.mkdir(parents=True, exist_ok=True)
        th.save(self.network.state_dict(), agent_file_path.joinpath(agent_main_filename))

    def load(self, filename: str) -> None:
        """
        Load a saved network state dict from the training package's ``saved_agents`` directory.

        The checkpoint is mapped onto the network's current device, so a state dict saved on one
        device (e.g. ``cuda``) round-trips onto a network on another (e.g. ``cpu``).

        :param filename: name of the file under which the network state dict was saved
        """
        agent_file_path = pathlib.Path(__file__).parent.joinpath("saved_agents/")
        state_dict = th.load(
            agent_file_path.joinpath(filename), map_location=self.network.device, weights_only=True,
        )
        self.network.load_state_dict(state_dict)
