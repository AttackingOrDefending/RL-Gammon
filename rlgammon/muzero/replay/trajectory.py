"""Trajectory data contracts for the Stochastic MuZero replay buffer.

A :class:`Trajectory` is the record of a single self-play game: an ordered list of
:class:`Step` objects, one per decision node visited, plus the per-player terminal returns. The
replay buffer slices unroll windows out of these trajectories and turns them into the training
:class:`~rlgammon.muzero.replay.replay_buffer.Batch`. The :class:`UnrollTarget` bundles the
per-window targets (values, rewards, dense policies and following observations) that WU-D4 consumes.
"""
from dataclasses import dataclass, field

from rlgammon.muzero.muzero_types import MuZeroConfig


@dataclass
class Step:
    """A single recorded decision step of a self-play game (one decision node)."""

    observation: list[float]
    action: int
    reward: float
    policy: dict[int, float]
    value: float
    to_play: int


@dataclass
class UnrollTarget:
    """
    The per-window training targets sliced from a trajectory for a single unroll start index.

    All lists have length ``K + 1`` (with ``K = config.unroll_steps``), except ``actions`` and
    ``following_observations`` which have length ``K`` (one per unrolled transition).

    :param observation: the root observation of the window (the side-to-move features at the start).
    :param actions: the ``K`` actions unrolled from the root.
    :param target_values: the ``K + 1`` n-step bootstrapped return targets for steps ``t .. t + K``.
    :param target_rewards: the ``K + 1`` reward targets for steps ``t .. t + K``.
    :param target_policies: the ``K + 1`` dense policy targets, each of length ``num_actions``.
    :param following_observations: the ``K`` real observations following each unrolled action.
    """

    observation: list[float]
    actions: list[int]
    target_values: list[float]
    target_rewards: list[float]
    target_policies: list[list[float]]
    following_observations: list[list[float]]


@dataclass
class Trajectory:
    """The full record of one self-play game: a list of decision steps and the terminal returns."""

    steps: list[Step] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        """
        Return the number of recorded decision steps.

        :return: the number of :class:`Step` objects in the trajectory
        """
        return len(self.steps)

    def make_target(self, start_index: int, config: MuZeroConfig) -> UnrollTarget:
        """
        Build the unroll targets for a window of ``unroll_steps`` starting at ``start_index``.

        For every step ``t`` in ``start_index .. start_index + unroll_steps`` the value target is the
        ``td_steps``-step bootstrapped return: the discounted sum of the rewards over the next
        ``td_steps`` transitions plus the stored MCTS root value at ``t + td_steps`` discounted by
        ``discount ** td_steps`` (the bootstrap is dropped once it reaches or passes the end of the
        game, leaving the pure discounted reward-to-go). Steps at or beyond the end of the game are
        absorbing: their value and reward targets are zero and their policy target is uniform over
        every action. The actions, following observations and the root observation are read directly
        from the recorded steps, repeating the final observation once the window runs past the game.

        :param start_index: the index of the root step of the unroll window
        :param config: the configuration providing ``unroll_steps``, ``td_steps``, ``discount`` and ``num_actions``
        :return: the :class:`UnrollTarget` bundling the per-window value, reward, policy and observation targets
        """
        num_steps = len(self.steps)
        uniform_policy = [1.0 / config.num_actions] * config.num_actions

        root_observation = self._observation_at(start_index)
        actions: list[int] = []
        following_observations: list[list[float]] = []
        target_values: list[float] = []
        target_rewards: list[float] = []
        target_policies: list[list[float]] = []

        for offset in range(config.unroll_steps + 1):
            current = start_index + offset
            target_values.append(self._compute_value_target(current, config))
            if current < num_steps:
                target_rewards.append(self.steps[current].reward)
                target_policies.append(self._dense_policy(self.steps[current].policy, config.num_actions))
            else:
                target_rewards.append(0.0)
                target_policies.append(list(uniform_policy))
            # The K actions / following observations describe the transitions, so skip the final step.
            if offset < config.unroll_steps:
                if current < num_steps:
                    actions.append(self.steps[current].action)
                    following_observations.append(self._observation_at(current + 1))
                else:
                    actions.append(0)
                    following_observations.append(self._observation_at(current))

        return UnrollTarget(
            observation=root_observation,
            actions=actions,
            target_values=target_values,
            target_rewards=target_rewards,
            target_policies=target_policies,
            following_observations=following_observations,
        )

    def _compute_value_target(self, index: int, config: MuZeroConfig) -> float:
        """
        Compute the ``td_steps``-step bootstrapped return target for the step at ``index``.

        The target is expressed from the perspective of the player to move at ``index``. In a
        two-player game the player to move ALTERNATES every decision ply, while each stored
        :class:`Step` records its value (the MCTS root estimate) and its reward (the terminal return,
        non-zero only on the last step) from THAT step's own mover's perspective. Both the bootstrap
        value at ``index + td_steps`` and every folded reward are therefore sign-flipped when the step
        they come from belongs to the opponent of ``index``'s mover, so the whole target is a single,
        consistent perspective. A missing flip points the value target the wrong way and cripples
        learning (the agent learns to maximise the opponent's return on alternating plies).

        :param index: the step index whose value target to compute (may lie past the game's end)
        :param config: the configuration providing ``td_steps``, ``discount`` and the step count
        :return: the bootstrapped return from ``index``'s mover's perspective, or ``0.0`` past terminal
        """
        num_steps = len(self.steps)
        if index >= num_steps:
            return 0.0
        root_to_play = self.steps[index].to_play
        bootstrap_index = index + config.td_steps
        if bootstrap_index < num_steps:
            bootstrap_sign = 1.0 if self.steps[bootstrap_index].to_play == root_to_play else -1.0
            value = bootstrap_sign * self.steps[bootstrap_index].value * config.discount**config.td_steps
        else:
            value = 0.0
        for reward_index in range(index, min(bootstrap_index, num_steps)):
            reward_sign = 1.0 if self.steps[reward_index].to_play == root_to_play else -1.0
            value += reward_sign * self.steps[reward_index].reward * config.discount ** (reward_index - index)
        return value

    def _observation_at(self, index: int) -> list[float]:
        """
        Return a copy of the observation at ``index``, clamped to the last recorded observation.

        :param index: the step index whose observation to return (clamped into ``[0, len - 1]``)
        :return: a copy of the recorded observation list at the clamped index
        """
        clamped = min(index, len(self.steps) - 1)
        return list(self.steps[clamped].observation)

    @staticmethod
    def _dense_policy(policy: dict[int, float], num_actions: int) -> list[float]:
        """
        Expand a sparse action -> probability mapping into a dense vector over all actions.

        :param policy: the sparse (normalized) visit-count distribution keyed by action id
        :param num_actions: the size of the dense action space
        :return: a length-``num_actions`` list with the sparse mass scattered into place
        """
        dense = [0.0] * num_actions
        for action, probability in policy.items():
            dense[action] = probability
        return dense
