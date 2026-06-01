"""Tree-node primitives for the Stochastic MuZero search.

The search alternates between two node kinds along every turn:

* a :class:`DecisionNode` whose children are keyed by environment action id, and
* a :class:`ChanceNode` whose children are keyed by chance-codebook index.

Both kinds are defined in this single module because their child types reference each other
(``DecisionNode.children`` holds :class:`ChanceNode` values and vice versa); the mutual reference
resolves cleanly because the child-type annotations live on instance attributes inside ``__init__``
and are therefore never evaluated at runtime (PEP 526), so no ``from __future__`` import is needed.
:class:`MinMaxStats` tracks the running value bounds used to normalize the exploitation term of the
predictor + upper-confidence-bound (pUCT) selection rule.
"""
import math

import numpy as np
import torch as th


class MinMaxStats:
    """Track the minimum and maximum backed-up values seen, to normalize Q values into ``[0, 1]``."""

    def __init__(self) -> None:
        """Initialise the bounds to an empty (inverted) interval so the first update sets both ends."""
        self.maximum = -math.inf
        self.minimum = math.inf

    def update(self, value: float) -> None:
        """
        Widen the tracked interval to include ``value``.

        :param value: a backed-up value to fold into the running min/max bounds
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """
        Scale ``value`` into ``[0, 1]`` using the tracked bounds.

        While fewer than two distinct values have been seen (so the interval is empty or degenerate)
        the value is returned unchanged, matching the canonical MuZero behaviour.

        :param value: the value to normalize
        :return: the normalized value, or ``value`` itself when the bounds are not yet usable
        """
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class DecisionNode:
    """A decision node whose children, keyed by action id, are :class:`ChanceNode` instances."""

    def __init__(self, prior: float) -> None:
        """
        Construct an unexpanded decision node carrying its selection prior.

        :param prior: the prior probability of reaching this node from its parent chance node
        """
        self.prior = prior
        self.to_play = 0
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.state: th.Tensor | None = None
        self.children: dict[int, ChanceNode] = {}

    def value(self) -> float:
        """
        Return the mean backed-up value of this node.

        :return: ``value_sum / visit_count`` or ``0.0`` when the node has never been visited
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """
        Report whether this node already has children.

        :return: ``True`` if the node has been expanded with child chance nodes
        """
        return len(self.children) > 0

    def expand(self, state: th.Tensor, reward: float, to_play: int, policy_logits: th.Tensor,
               legal_actions: list[int] | None, max_children: int | None = None) -> None:
        """
        Expand the node by creating one child :class:`ChanceNode` per (kept) action.

        Priors are the softmax of ``policy_logits``. When ``legal_actions`` is provided the children
        are restricted to those actions. When ``max_children`` is provided only the top-``max_children``
        actions by prior are kept (a cheap top-k restriction that keeps the per-node fan-out small over
        a large action space, which dominates the search cost at deep internal nodes); the kept priors
        are always renormalized to sum to one.

        :param state: the latent state produced for this node by the network
        :param reward: the scalar reward of the transition that produced this node
        :param to_play: the player to move at this node
        :param policy_logits: policy logits of shape ``[num_actions]`` over all environment actions
        :param legal_actions: the actions to restrict children to, or ``None`` to allow every action
        :param max_children: keep only the top-``max_children`` actions by prior, or ``None`` for all
        """
        self.state = state
        self.reward = reward
        self.to_play = to_play
        priors = self._action_priors(policy_logits, legal_actions, max_children)
        total = sum(priors.values())
        if total <= 0.0:
            # Degenerate logits (e.g. masked support): fall back to a uniform prior over the actions.
            uniform = 1.0 / len(priors) if priors else 0.0
            priors = dict.fromkeys(priors, uniform)
            total = 1.0
        for action, prior in priors.items():
            self.children[action] = ChanceNode(prior / total)

    @staticmethod
    def _action_priors(policy_logits: th.Tensor, legal_actions: list[int] | None,
                       max_children: int | None) -> dict[int, float]:
        """
        Build the (optionally top-k restricted) action -> softmax-prior mapping in a vectorised way.

        The softmax and any top-k selection happen on-device over tensors, and only the small set of
        KEPT priors is moved to the host with a single transfer. This avoids the per-action
        ``float(policy[action])`` device-to-host syncs that otherwise dominate the search over a large
        (1352) action space.

        :param policy_logits: policy logits of shape ``[num_actions]`` over all environment actions
        :param legal_actions: the actions to restrict to, or ``None`` to allow every action
        :param max_children: keep only the top-``max_children`` actions by prior, or ``None`` for all
        :return: an ordered ``action -> prior`` mapping over the kept actions
        """
        policy = th.softmax(policy_logits, dim=0)
        if legal_actions is not None:
            index = th.tensor(legal_actions, dtype=th.long, device=policy.device)
            masked = policy.index_select(0, index)
            if max_children is not None and masked.shape[0] > max_children:
                top_values, top_positions = th.topk(masked, max_children)
                kept_actions = index.index_select(0, top_positions).tolist()
                return dict(zip(kept_actions, top_values.tolist(), strict=True))
            return dict(zip(legal_actions, masked.tolist(), strict=True))
        if max_children is not None and policy.shape[0] > max_children:
            top_values, top_actions = th.topk(policy, max_children)
            return dict(zip(top_actions.tolist(), top_values.tolist(), strict=True))
        return dict(enumerate(policy.tolist()))

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float,
                              rng: np.random.Generator) -> None:
        """
        Mix Dirichlet noise into the children's priors to encourage root exploration.

        :param dirichlet_alpha: the concentration parameter of the Dirichlet distribution
        :param exploration_fraction: the fraction of the prior mass replaced by the sampled noise
        :param rng: the random number generator used to sample the Dirichlet noise
        """
        actions = list(self.children)
        if not actions:
            return
        noise = rng.dirichlet([dirichlet_alpha] * len(actions))
        for action, sampled in zip(actions, noise, strict=True):
            child = self.children[action]
            child.prior = child.prior * (1.0 - exploration_fraction) + float(sampled) * exploration_fraction


class ChanceNode:
    """A chance node whose children, keyed by codebook index, are :class:`DecisionNode` instances."""

    def __init__(self, prior: float) -> None:
        """
        Construct an unexpanded chance node carrying its selection prior.

        :param prior: the prior probability of reaching this node (the parent action's policy mass)
        """
        self.prior = prior
        self.to_play = 0
        self.visit_count = 0
        self.value_sum = 0.0
        self.afterstate: th.Tensor | None = None
        self.children: dict[int, DecisionNode] = {}

    def value(self) -> float:
        """
        Return the mean backed-up value of this node.

        :return: ``value_sum / visit_count`` or ``0.0`` when the node has never been visited
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """
        Report whether this node already has children.

        :return: ``True`` if the node has been expanded with child decision nodes
        """
        return len(self.children) > 0

    def expand(self, afterstate: th.Tensor, to_play: int, sigma: th.Tensor) -> None:
        """
        Expand the node by creating one child :class:`DecisionNode` per codebook index.

        The child for codebook index ``c`` is given prior ``sigma[c]`` (the predicted chance
        distribution). Each child decision node belongs to the opponent of ``to_play``.

        :param afterstate: the afterstate latent produced for this node by the network
        :param to_play: the player to move at this (afterstate) node, equal to the parent's player
        :param sigma: the chance prior of shape ``[codebook_size]`` summing to one
        """
        self.afterstate = afterstate
        self.to_play = to_play
        # One ``tolist()`` host transfer of the whole codebook, rather than a ``float(sigma[c])``
        # device-to-host sync per code (which dominates the search over many chance expansions).
        for code, prior in enumerate(sigma.tolist()):
            self.children[code] = DecisionNode(prior)
