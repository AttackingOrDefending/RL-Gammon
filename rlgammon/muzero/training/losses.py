"""The K-step unrolled training loss of the Stochastic MuZero agent.

The loss unrolls the learned model from the root observation of every batch window for
``config.unroll_steps`` (``K``) steps, alternating the afterstate model (the deterministic action
transition) and the dynamics model (the stochastic chance transition), and accumulates a categorical
cross-entropy loss against the targets stored in the
:class:`~rlgammon.muzero.replay.replay_buffer.Batch`.

Index alignment (confirmed against :meth:`~rlgammon.muzero.replay.trajectory.Trajectory.make_target`,
with ``K = config.unroll_steps`` and ``t`` the window's root step):

* ``target_values[:, j]`` / ``target_policies[:, j]`` for ``j`` in ``0 .. K`` are the value and dense
  policy targets of step ``t + j``.
* ``target_rewards[:, j]`` for ``j`` in ``0 .. K`` is the recorded reward of step ``t + j``.
* ``actions[:, k]`` for ``k`` in ``0 .. K - 1`` is the action played at step ``t + k``.
* ``chance_observations[:, k]`` for ``k`` in ``0 .. K - 1`` is the real observation that follows the
  action at step ``t + k`` (the next decision node's observation); it is encoded by the chance encoder
  to supply the chance class target and the straight-through one-hot fed to the dynamics.

The per-step losses are therefore:

* step 0 (root, from ``initial_inference``): value CE at ``target_values[:, 0]`` and policy CE at
  ``target_policies[:, 0]``;
* for each recurrent step ``k`` in ``0 .. K - 1`` (afterstate then state): the afterstate value CE
  shares the step's value target ``target_values[:, k]``, the chance CE is between the afterstate
  ``chance_logits`` and the encoded chance class, the dynamics reward CE is at ``target_rewards[:, k]``
  and the resulting state's value / policy CE are at ``target_values[:, k + 1]`` /
  ``target_policies[:, k + 1]``.

Gradient scaling: every loss term raised at recurrent step ``k`` (afterstate value, chance, reward,
value and policy at ``k + 1``, and the VQ regularizer) is scaled by ``1 / K`` so the recurrent unroll
contributes on the same scale as the single root step, mirroring the MuZero gradient-scaling recipe.
The VQ regularizer (commitment + codebook diversity) is already multiplied by its costs
(``config.commitment_cost`` / ``config.codebook_diversity_cost``) inside
:meth:`~rlgammon.muzero.networks.chance_encoder.ChanceEncoder.forward`, so it is only gradient-scaled
here and never re-weighted. It is reported under the ``"commitment"`` loss key for continuity.
"""
import torch as th

from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.networks.value_encoding import scalar_to_support
from rlgammon.muzero.replay.replay_buffer import Batch


def _scalar_cross_entropy(logits: th.Tensor, scalar_target: th.Tensor, support_size: int) -> th.Tensor:
    """
    Categorical cross-entropy between predicted logits and a two-hot encoded scalar target.

    The scalar target is encoded to a two-hot probability distribution over the support atoms and the
    standard categorical cross-entropy ``-(target * log_softmax(logits)).sum(-1)`` is returned per row.

    :param logits: predicted categorical logits of shape ``[B, support_size]``
    :param scalar_target: scalar target of shape ``[B]`` to two-hot encode
    :param support_size: number of categorical atoms of the support
    :return: per-row cross-entropy of shape ``[B]``
    """
    target = scalar_to_support(scalar_target, support_size)
    return -(target * th.log_softmax(logits, dim=1)).sum(dim=1)


def _distribution_cross_entropy(logits: th.Tensor, target: th.Tensor) -> th.Tensor:
    """
    Categorical cross-entropy between predicted logits and a target probability distribution.

    :param logits: predicted categorical logits of shape ``[B, C]``
    :param target: target probability distribution of shape ``[B, C]`` (rows summing to one)
    :return: per-row cross-entropy of shape ``[B]``
    """
    return -(target * th.log_softmax(logits, dim=1)).sum(dim=1)


def _one_hot(indices: th.Tensor, num_classes: int) -> th.Tensor:
    """
    Build a float one-hot encoding of a batch of class indices.

    :param indices: long tensor of shape ``[B]`` holding the active class per row
    :param num_classes: width of the produced one-hot vectors
    :return: float one-hot tensor of shape ``[B, num_classes]``
    """
    return th.nn.functional.one_hot(indices, num_classes=num_classes).to(th.float32)


def muzero_loss(config: MuZeroConfig, network: StochasticMuZeroNetwork,
                batch: Batch) -> dict[str, th.Tensor]:
    """
    Compute the K-step unrolled Stochastic MuZero training loss for a batch of unroll windows.

    The model is unrolled from each window's root observation for ``config.unroll_steps`` steps and the
    categorical value, policy, reward and chance losses are accumulated with the index alignment and
    ``1 / K`` recurrent gradient scaling documented at the module level. Every per-row loss is weighted
    by ``batch.weights`` and averaged over the batch.

    :param config: the configuration providing the unroll length, support sizes and loss weights
    :param network: the Stochastic MuZero network providing every inference entry point
    :param batch: the stacked batch of unroll windows produced by the replay buffer
    :return: a dict with keys ``{"total", "value", "policy", "reward", "chance", "commitment"}``, each a
        scalar tensor, where ``total`` is the configured weighted sum of the component losses
    """
    unroll_steps = config.unroll_steps
    num_actions = config.num_actions
    value_support = config.value_support_size
    reward_support = config.reward_support_size
    weights = batch.weights

    # Recurrent-step gradient scaling so the K unrolled steps contribute on the root step's scale.
    recurrent_scale = 1.0 / unroll_steps if unroll_steps > 0 else 1.0

    out0 = network.initial_inference(batch.observation)
    value_loss = _scalar_cross_entropy(out0.value, batch.target_values[:, 0], value_support)
    policy_loss = _distribution_cross_entropy(out0.policy_logits, batch.target_policies[:, 0])
    reward_loss = th.zeros_like(value_loss)
    chance_loss = th.zeros_like(value_loss)
    # The VQ regularizer (commitment + codebook diversity) is returned per step as a scalar (already
    # averaged over the batch by the chance encoder), so it is accumulated as a scalar rather than as
    # a per-row tensor. It is reported under the ``"commitment"`` key for log continuity.
    commitment = out0.value.new_zeros(())

    state = out0.state
    for step in range(unroll_steps):
        action_onehot = _one_hot(batch.actions[:, step], num_actions)
        afterstate_output = network.recurrent_inference_afterstate(state, action_onehot)

        # The chance target is the encoding of the real next observation; the straight-through one-hot
        # is reused as the dynamics input so the codes and gradients stay consistent.
        chance_onehot, code_indices, commitment_step = network.encode_chance(
            batch.chance_observations[:, step],
        )

        afterstate_value_loss = _scalar_cross_entropy(
            afterstate_output.afterstate_value, batch.target_values[:, step], value_support,
        )
        chance_step_loss = th.nn.functional.cross_entropy(
            afterstate_output.chance_logits, code_indices, reduction="none",
        )

        next_output = network.recurrent_inference_state(afterstate_output.afterstate, chance_onehot)
        reward_step_loss = _scalar_cross_entropy(
            next_output.reward, batch.target_rewards[:, step], reward_support,
        )
        value_step_loss = _scalar_cross_entropy(
            next_output.value, batch.target_values[:, step + 1], value_support,
        )
        policy_step_loss = _distribution_cross_entropy(
            next_output.policy_logits, batch.target_policies[:, step + 1],
        )

        value_loss = value_loss + recurrent_scale * (afterstate_value_loss + value_step_loss)
        policy_loss = policy_loss + recurrent_scale * policy_step_loss
        reward_loss = reward_loss + recurrent_scale * reward_step_loss
        chance_loss = chance_loss + recurrent_scale * chance_step_loss
        # ``commitment_step`` (commitment + codebook-diversity, already weighted by their costs inside
        # ``encode_chance``) is only gradient-scaled here and never re-weighted.
        commitment = commitment + recurrent_scale * commitment_step

        state = next_output.state

    value = _weighted_mean(value_loss, weights)
    policy = _weighted_mean(policy_loss, weights)
    reward = _weighted_mean(reward_loss, weights)
    chance = _weighted_mean(chance_loss, weights)

    total = (
        config.value_loss_weight * value
        + config.policy_loss_weight * policy
        + config.reward_loss_weight * reward
        + config.chance_loss_weight * chance
        + commitment
    )

    return {
        "total": total,
        "value": value,
        "policy": policy,
        "reward": reward,
        "chance": chance,
        "commitment": commitment,
    }


def _weighted_mean(per_row_loss: th.Tensor, weights: th.Tensor) -> th.Tensor:
    """
    Reduce a per-row loss to a scalar by an importance-weighted mean over the batch.

    :param per_row_loss: per-row loss of shape ``[B]``
    :param weights: per-row importance weights of shape ``[B]``
    :return: the scalar weighted mean ``mean(weights * per_row_loss)``
    """
    return th.mean(weights * per_row_loss)
