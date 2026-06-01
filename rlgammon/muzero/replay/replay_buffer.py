"""Uniform replay buffer producing training batches for the Stochastic MuZero agent.

The buffer stores whole self-play :class:`~rlgammon.muzero.replay.trajectory.Trajectory` objects and
caps the total number of stored decision steps at ``config.replay_capacity``, dropping the oldest
trajectories first. :meth:`MuZeroReplayBuffer.sample_batch` draws ``config.batch_size`` unroll
windows (a trajectory then a start index within it), turns each into the per-window targets via
:meth:`~rlgammon.muzero.replay.trajectory.Trajectory.make_target` and stacks them into a
:class:`Batch` of torch tensors with the frozen shapes consumed by the trainer (WU-D4).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch as th

from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.replay.trajectory import Trajectory, UnrollTarget


@dataclass
class Batch:
    """
    A stacked batch of unroll windows ready for a Stochastic MuZero training step.

    With ``B = config.batch_size``, ``K = config.unroll_steps``, ``A = config.num_actions`` and
    ``O = config.observation_size`` the fields have the following fixed shapes:

    :param observation: ``[B, O]`` root observation of each unroll window.
    :param actions: ``[B, K]`` (long) the ``K`` actions unrolled from each root.
    :param target_values: ``[B, K + 1]`` n-step bootstrapped return targets for steps ``t .. t + K``.
    :param target_rewards: ``[B, K + 1]`` reward targets for steps ``t .. t + K``.
    :param target_policies: ``[B, K + 1, A]`` dense policy targets (uniform for absorbing steps).
    :param chance_observations: ``[B, K, O]`` real observations following each unrolled action.
    :param weights: ``[B]`` importance weights (all ones for uniform sampling).
    """

    observation: th.Tensor
    actions: th.Tensor
    target_values: th.Tensor
    target_rewards: th.Tensor
    target_policies: th.Tensor
    chance_observations: th.Tensor
    weights: th.Tensor


class MuZeroReplayBuffer:
    """A FIFO replay buffer over self-play trajectories with uniform unroll-window sampling."""

    def __init__(self, config: MuZeroConfig) -> None:
        """
        Construct an empty buffer bound to a configuration.

        :param config: the configuration providing ``replay_capacity``, ``batch_size``, ``unroll_steps``,
            ``td_steps``, ``discount``, ``num_actions`` and ``observation_size``
        """
        self.config = config
        self._trajectories: deque[Trajectory] = deque()
        self._total_steps = 0

    def __len__(self) -> int:
        """
        Return the total number of decision steps currently stored across all trajectories.

        :return: the summed length of every stored trajectory
        """
        return self._total_steps

    def save(self, trajectory: Trajectory) -> None:
        """
        Append a trajectory and evict the oldest ones until the step capacity is respected.

        Empty trajectories are ignored. A single trajectory larger than the capacity is still kept
        (it is the most recent data) but every older trajectory is dropped.

        :param trajectory: the finished self-play trajectory to store
        """
        if len(trajectory) == 0:
            return
        self._trajectories.append(trajectory)
        self._total_steps += len(trajectory)
        while self._total_steps > self.config.replay_capacity and len(self._trajectories) > 1:
            evicted = self._trajectories.popleft()
            self._total_steps -= len(evicted)

    def sample_batch(self, rng: np.random.Generator) -> Batch:
        """
        Sample ``config.batch_size`` unroll windows uniformly and stack them into a :class:`Batch`.

        Each sample picks a trajectory uniformly at random, then a start index uniformly within it,
        and builds the ``K + 1`` targets (n-step bootstrapped values, rewards, dense policies) and the
        ``K`` following observations via :meth:`Trajectory.make_target`. All windows are stacked into
        torch tensors of the frozen :class:`Batch` shapes.

        :param rng: the random number generator driving trajectory and start-index selection
        :return: a :class:`Batch` of ``config.batch_size`` stacked unroll windows
        :raises ValueError: if the buffer is empty
        """
        if not self._trajectories:
            raise ValueError(_EMPTY_BUFFER_ERROR)

        targets = [self._sample_target(rng) for _ in range(self.config.batch_size)]

        # Stacking is done on the CPU (the targets are python lists) then the whole batch is moved to
        # the configured device in one transfer so the heavy batched training runs on-device.
        device = th.device(self.config.device)
        observation = th.tensor([target.observation for target in targets], dtype=th.float32, device=device)
        actions = th.tensor([target.actions for target in targets], dtype=th.long, device=device)
        target_values = th.tensor([target.target_values for target in targets], dtype=th.float32, device=device)
        target_rewards = th.tensor([target.target_rewards for target in targets], dtype=th.float32, device=device)
        target_policies = th.tensor([target.target_policies for target in targets], dtype=th.float32, device=device)
        chance_observations = th.tensor(
            [target.following_observations for target in targets], dtype=th.float32, device=device,
        )
        weights = th.ones(self.config.batch_size, dtype=th.float32, device=device)

        return Batch(
            observation=observation,
            actions=actions,
            target_values=target_values,
            target_rewards=target_rewards,
            target_policies=target_policies,
            chance_observations=chance_observations,
            weights=weights,
        )

    def _sample_target(self, rng: np.random.Generator) -> UnrollTarget:
        """
        Sample one trajectory and start index and build its unroll target.

        :param rng: the random number generator driving the selection
        :return: the :class:`UnrollTarget` for the sampled window
        """
        trajectory = self._trajectories[int(rng.integers(len(self._trajectories)))]
        start_index = int(rng.integers(len(trajectory)))
        return trajectory.make_target(start_index, self.config)


# Raised when a batch is requested from a buffer that holds no trajectories.
_EMPTY_BUFFER_ERROR = "Cannot sample a batch from an empty replay buffer."
