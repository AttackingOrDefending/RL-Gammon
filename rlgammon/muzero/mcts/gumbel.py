"""Gumbel MuZero root primitives (Danihelka et al., 2022, "Policy improvement by planning with Gumbel").

These pure, network-free helpers implement the three ingredients of Gumbel MuZero's root action
selection that replace Dirichlet noise + visit-proportional sampling with a provably policy-improving,
low-simulation scheme:

* :func:`sample_gumbel` draws i.i.d. Gumbel(0) noise for the Gumbel-top-k trick.
* :func:`sequential_halving_schedule` splits a fixed simulation budget across the
  ``ceil(log2(m))`` halving phases, assigning each surviving action an equal share per phase.
* :func:`sigma` is the monotonic Q-transform applied to the (completed) action values, and
  :func:`completed_q_values` fills the Q of unvisited actions with the value-weighted prior mean so
  every action has a value estimate.
* :func:`gumbel_improved_policy` turns the prior logits and completed Q values into the improved
  policy ``softmax(logits + sigma(completed_q))`` used as the stored policy target.

All functions operate on a SINGLE root (1-D tensors over the considered/legal actions); the batched
self-play engine calls them per tree. The Gumbel-argmax selection itself (``argmax_a g(a) + logit(a)
+ sigma(q(a))`` over the survivors) is performed by the caller, which owns the per-action statistics.
"""
import math

import numpy as np
import torch as th

# Default Gumbel "visit scale" and value scale constants from the paper's reference configuration.
# ``c_visit`` and ``c_scale`` enter sigma(q) = (c_visit + max_visit) * c_scale * q.
DEFAULT_C_VISIT = 50.0
DEFAULT_C_SCALE = 1.0
# Smallest uniform sample admitted before taking the double log, keeping the Gumbel transform finite.
_UNIFORM_FLOOR = 1e-12


def sample_gumbel(num_actions: int, rng: np.random.Generator) -> th.Tensor:
    """
    Sample a vector of i.i.d. standard Gumbel(0, 1) variates for the Gumbel-top-k trick.

    :param num_actions: the number of variates to draw (one per candidate action)
    :param rng: the random number generator to draw the underlying uniforms from
    :return: a float tensor of shape ``[num_actions]`` of Gumbel(0) samples
    """
    # Gumbel(0) = -log(-log(U)) with U ~ Uniform(0, 1); clamp away from 0 to keep the logs finite.
    uniforms = rng.random(num_actions)
    uniforms = np.clip(uniforms, _UNIFORM_FLOOR, 1.0)
    gumbel = -np.log(-np.log(uniforms))
    return th.tensor(gumbel, dtype=th.float32)


def sequential_halving_schedule(num_simulations: int, num_considered: int) -> list[int]:
    """
    Compute the per-phase simulation count each surviving action receives under sequential halving.

    The budget is split across ``ceil(log2(num_considered))`` phases. In each phase every currently
    surviving action is simulated an equal number of times (at least once), the set is halved by the
    Gumbel-argmax score, and the next phase gets the remaining budget. The returned list holds the
    per-action simulation count of each phase (its length is the number of phases).

    :param num_simulations: the total simulation budget for the root
    :param num_considered: the number of actions initially considered (``m``, a power of two or not)
    :return: a list whose entry ``p`` is the number of simulations each survivor gets in phase ``p``
    """
    if num_considered <= 1 or num_simulations <= 0:
        return [num_simulations] if num_simulations > 0 else []
    num_phases = max(1, math.ceil(math.log2(num_considered)))
    schedule: list[int] = []
    remaining_actions = num_considered
    remaining_sims = num_simulations
    for phase in range(num_phases):
        phases_left = num_phases - phase
        # Even split of the remaining budget across the remaining phases, then across the survivors.
        budget_this_phase = remaining_sims // phases_left
        per_action = max(1, budget_this_phase // remaining_actions)
        schedule.append(per_action)
        remaining_sims -= per_action * remaining_actions
        remaining_actions = max(1, remaining_actions // 2)
    return schedule


def sigma(q_values: th.Tensor, max_visit: int, *, c_visit: float = DEFAULT_C_VISIT,
          c_scale: float = DEFAULT_C_SCALE) -> th.Tensor:
    """
    Apply the monotonic Gumbel-MuZero Q-transform ``sigma(q) = (c_visit + max_visit) * c_scale * q``.

    The transform scales the (already normalized) action values by the maximum child visit count so
    that, as more simulations accumulate, the Q term increasingly dominates the prior logits in the
    selection score, matching the paper's recommended monotonic transform.

    :param q_values: the (completed, normalized) action values of shape ``[num_actions]``
    :param max_visit: the maximum visit count over the root's children
    :param c_visit: the additive visit constant of the transform
    :param c_scale: the multiplicative scale of the transform
    :return: the transformed values of shape ``[num_actions]``
    """
    return (c_visit + float(max_visit)) * c_scale * q_values


def completed_q_values(prior: th.Tensor, raw_q: th.Tensor, visit_counts: th.Tensor,
                       value_prior: float) -> th.Tensor:
    """
    Complete the action-value vector by filling unvisited actions with the prior value estimate.

    Following the paper, visited actions keep their empirical mean value while every unvisited action
    is assigned the "value approximation" ``value_prior`` (the prior-weighted mean of the visited
    children's values, computed by the caller; the root network value is a fine stand-in). This gives
    every action a value so :func:`gumbel_improved_policy` and the Gumbel-argmax are well defined.

    :param prior: the (softmax) prior over actions of shape ``[num_actions]`` (unused here but kept
        for a symmetric signature with the paper's value approximation; the caller passes the mean)
    :param raw_q: the empirical mean value per action of shape ``[num_actions]`` (0 where unvisited)
    :param visit_counts: the visit count per action of shape ``[num_actions]``
    :param value_prior: the scalar value approximation used for unvisited actions
    :return: the completed action values of shape ``[num_actions]``
    """
    del prior
    visited = visit_counts > 0
    return th.where(visited, raw_q, th.full_like(raw_q, value_prior))


def gumbel_improved_policy(logits: th.Tensor, completed_q: th.Tensor, max_visit: int, *,
                           c_visit: float = DEFAULT_C_VISIT,
                           c_scale: float = DEFAULT_C_SCALE) -> th.Tensor:
    """
    Compute the Gumbel-improved policy ``softmax(logits + sigma(completed_q))`` over the actions.

    This is the policy-improvement target of Gumbel MuZero: the prior logits shifted by the monotonic
    transform of the completed action values. It is used as the policy target stored in the trajectory
    (a strictly stronger improvement operator than visit-count normalization at low simulation counts).

    :param logits: the prior policy logits over the considered actions of shape ``[num_actions]``
    :param completed_q: the completed action values of shape ``[num_actions]``
    :param max_visit: the maximum visit count over the root's children (drives the sigma scale)
    :param c_visit: the additive visit constant of the sigma transform
    :param c_scale: the multiplicative scale of the sigma transform
    :return: the improved policy distribution of shape ``[num_actions]`` summing to one
    """
    transformed = logits + sigma(completed_q, max_visit, c_visit=c_visit, c_scale=c_scale)
    return th.softmax(transformed, dim=0)
