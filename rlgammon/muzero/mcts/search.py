"""Stochastic MuZero Monte-Carlo tree search running purely on the learned model.

This implements the search of "Planning in Stochastic Environments with a Learned Model"
(Antonoglou et al., 2022). The tree alternates :class:`DecisionNode` and :class:`ChanceNode`
plies: selecting an action at a decision node calls the afterstate dynamics to reach a chance
node, and selecting a chance outcome calls the dynamics to reach the next decision node (owned by
the opponent). Selection uses the predictor + upper-confidence-bound (pUCT) rule at both node
kinds, expansion runs the network in ``eval`` mode under :func:`torch.no_grad`, and the two-player
backup flips the sign of the value for nodes belonging to the opponent of the leaf's player.

The search is model-only: it consumes a root observation tensor and a list of legal action ids and
never touches the real game.
"""
import math

import numpy as np
import torch as th

from rlgammon.muzero.mcts.node import ChanceNode, DecisionNode, MinMaxStats
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork


class StochasticMuZeroMCTS:
    """Run Stochastic MuZero search on a learned model and return the root action visit counts."""

    def __init__(self, config: MuZeroConfig, network: StochasticMuZeroNetwork,
                 rng: np.random.Generator | None = None) -> None:
        """
        Construct the search around a configuration and a (frozen) learned network.

        :param config: the configuration providing the pUCT, exploration and simulation settings
        :param network: the learned Stochastic MuZero network providing all inference entry points
        :param rng: the random number generator for Dirichlet noise (defaults to a fresh generator)
        """
        self.config = config
        self.network = network
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        # The device every search tensor (root observation, action/codebook one-hots) is created on.
        # Taken from the config so the search stays device-agnostic and matches the network's device.
        self._device = th.device(config.device)
        self._root: DecisionNode | None = None

    def run(self, root_observation: th.Tensor, legal_actions: list[int], *,
            add_exploration_noise: bool = True) -> dict[int, int]:
        """
        Run the configured number of simulations from the root and return its action visit counts.

        The root :class:`DecisionNode` is built from the network's initial inference, expanded over
        ``legal_actions`` only, and optionally perturbed with Dirichlet exploration noise. Each
        simulation selects a path down to an unexpanded leaf, expands it through the appropriate
        recurrent inference, and backs the leaf value up the path.

        :param root_observation: the root observation tensor of shape ``[observation_size]`` or ``[1, obs]``
        :param legal_actions: the legal action ids the root is restricted to
        :param add_exploration_noise: whether to add Dirichlet noise to the root priors
        :return: a mapping from root action id to its child visit count
        """
        self.network.eval()
        observation = root_observation if root_observation.dim() > 1 else root_observation.unsqueeze(0)
        # Inference runs on the search device; a CPU-built root observation is moved on-device here.
        observation = observation.to(self._device)

        with th.no_grad():
            output = self.network.initial_inference(observation)
            root_value = float(self.network.value_to_scalar(output.value)[0])

        root = DecisionNode(prior=0.0)
        root.expand(
            state=output.state[0],
            reward=0.0,
            to_play=0,
            policy_logits=output.policy_logits[0],
            legal_actions=legal_actions,
        )
        if add_exploration_noise:
            root.add_exploration_noise(
                self.config.dirichlet_alpha, self.config.exploration_fraction, self.rng,
            )
        self._root = root

        min_max_stats = MinMaxStats()
        # Seed the normalizer with the root estimate so the very first selections are well scaled.
        min_max_stats.update(root_value)

        for _ in range(self.config.num_simulations):
            self._simulate(root, min_max_stats)

        return {action: child.visit_count for action, child in root.children.items()}

    def _simulate(self, root: DecisionNode, min_max_stats: MinMaxStats) -> None:
        """
        Run a single simulation: descend to a leaf, expand it, and back up its value.

        :param root: the root decision node of the search tree
        :param min_max_stats: the running value normalizer shared across simulations
        """
        decision_node = root
        # The path is the sequence of nodes (decision/chance, alternating) visited this simulation.
        search_path: list[DecisionNode | ChanceNode] = [decision_node]

        # Descend through fully expanded decision -> chance -> decision ... plies.
        while True:
            action, chance_node = self._select_child_decision(decision_node, min_max_stats)
            search_path.append(chance_node)
            if not chance_node.is_expanded():
                # The chosen chance node is the leaf to expand via the afterstate dynamics.
                value = self._expand_chance(decision_node, action, chance_node)
                leaf_to_play = chance_node.to_play
                break

            code, next_decision = self._select_child_chance(chance_node, min_max_stats)
            search_path.append(next_decision)
            if not next_decision.is_expanded():
                # The chosen decision node is the leaf to expand via the (chance) dynamics.
                value = self._expand_decision(chance_node, code, next_decision)
                leaf_to_play = next_decision.to_play
                break

            decision_node = next_decision

        self._backpropagate(search_path, value, leaf_to_play, min_max_stats)

    def _expand_chance(self, parent: DecisionNode, action: int, chance_node: ChanceNode) -> float:
        """
        Expand a leaf chance node via the afterstate dynamics and return its bootstrap value.

        The afterstate (and thus the chance node) belongs to the same player as ``parent``.

        :param parent: the decision node whose selected action led to ``chance_node``
        :param action: the action id selected at ``parent``
        :param chance_node: the unexpanded chance node to populate
        :return: the scalar afterstate value, from the chance node's player's perspective
        """
        assert parent.state is not None
        action_onehot = self._action_onehot(action)
        with th.no_grad():
            afterstate_output = self.network.recurrent_inference_afterstate(
                parent.state.unsqueeze(0), action_onehot,
            )
            sigma = th.softmax(afterstate_output.chance_logits[0], dim=0)
            value = float(self.network.value_to_scalar(afterstate_output.afterstate_value)[0])
        chance_node.expand(
            afterstate=afterstate_output.afterstate[0],
            to_play=parent.to_play,
            sigma=sigma,
        )
        return value

    def _expand_decision(self, parent: ChanceNode, code: int, decision_node: DecisionNode) -> float:
        """
        Expand a leaf decision node via the dynamics and return its bootstrap value.

        The resulting decision node belongs to the opponent of ``parent`` (after the chance roll).

        :param parent: the chance node whose selected outcome led to ``decision_node``
        :param code: the codebook index selected at ``parent``
        :param decision_node: the unexpanded decision node to populate
        :return: the scalar value, from the decision node's player's perspective
        """
        assert parent.afterstate is not None
        chance_onehot = self._chance_onehot(code)
        with th.no_grad():
            output = self.network.recurrent_inference_state(
                parent.afterstate.unsqueeze(0), chance_onehot,
            )
            reward = float(self.network.reward_to_scalar(output.reward)[0])
            value = float(self.network.value_to_scalar(output.value)[0])
        decision_node.expand(
            state=output.state[0],
            reward=reward,
            to_play=1 - parent.to_play,
            policy_logits=output.policy_logits[0],
            legal_actions=None,
        )
        return value

    def _select_child_decision(self, node: DecisionNode,
                               min_max_stats: MinMaxStats) -> tuple[int, ChanceNode]:
        """
        Select an action at a decision node by the pUCT rule.

        :param node: the decision node to select an action at
        :param min_max_stats: the running value normalizer for the exploitation term
        :return: a tuple ``(action, child_chance_node)`` of the selected action and its child
        :raises RuntimeError: if the node has no children (it must be expanded before selection)
        """
        best_score = -math.inf
        best_action = -1
        best_child: ChanceNode | None = None
        c_puct = self._c_puct(node.visit_count)
        sqrt_total = math.sqrt(node.visit_count)
        for action, child in node.children.items():
            score = self._puct_score(child, c_puct, sqrt_total, node.to_play, min_max_stats)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise RuntimeError(_UNEXPANDED_SELECTION_ERROR)
        return best_action, best_child

    def _select_child_chance(self, node: ChanceNode,
                             min_max_stats: MinMaxStats) -> tuple[int, DecisionNode]:
        """
        Select a chance outcome at a chance node by the pUCT rule, using ``sigma`` as the prior.

        :param node: the chance node to select an outcome at
        :param min_max_stats: the running value normalizer for the exploitation term
        :return: a tuple ``(code, child_decision_node)`` of the selected codebook index and its child
        :raises RuntimeError: if the node has no children (it must be expanded before selection)
        """
        best_score = -math.inf
        best_code = -1
        best_child: DecisionNode | None = None
        c_puct = self._c_puct(node.visit_count)
        sqrt_total = math.sqrt(node.visit_count)
        for code, child in node.children.items():
            score = self._puct_score(child, c_puct, sqrt_total, node.to_play, min_max_stats)
            if score > best_score:
                best_score = score
                best_code = code
                best_child = child
        if best_child is None:
            raise RuntimeError(_UNEXPANDED_SELECTION_ERROR)
        return best_code, best_child

    def _puct_score(self, child: DecisionNode | ChanceNode, c_puct: float, sqrt_total: float,
                    parent_to_play: int, min_max_stats: MinMaxStats) -> float:
        """
        Compute the pUCT score of a child from the perspective of the player to move at the parent.

        The exploration term is ``P(a) * c_puct * sqrt(sum_b N_b) / (1 + N(a))``. The exploitation
        term is the normalized child mean value, negated when the child belongs to the opponent of
        the parent (the child's stored value is from the child's player's perspective); unvisited
        children contribute a zero exploitation term.

        :param child: the candidate child node
        :param c_puct: the pUCT coefficient derived from the parent visit count
        :param sqrt_total: the square root of the parent's visit count
        :param parent_to_play: the player to move at the parent node
        :param min_max_stats: the running value normalizer for the exploitation term
        :return: the pUCT score of the child
        """
        prior_score = child.prior * c_puct * sqrt_total / (1 + child.visit_count)
        if child.visit_count > 0:
            sign = 1.0 if child.to_play == parent_to_play else -1.0
            value_score = min_max_stats.normalize(sign * child.value())
        else:
            value_score = 0.0
        return value_score + prior_score

    def _c_puct(self, parent_visit_count: int) -> float:
        """
        Compute the visit-count-dependent pUCT coefficient.

        :param parent_visit_count: the parent node's visit count (``sum_b N_b``)
        :return: ``c_puct_init + log((sum_b N_b + c_puct_base + 1) / c_puct_base)``
        """
        return self.config.c_puct_init + math.log(
            (parent_visit_count + self.config.c_puct_base + 1) / self.config.c_puct_base,
        )

    def _backpropagate(self, search_path: list[DecisionNode | ChanceNode], value: float,
                       to_play: int, min_max_stats: MinMaxStats) -> None:
        """
        Back the leaf value up the search path with the two-player sign-flip and reward folding.

        Walking from the leaf to the root, each node accumulates ``+value`` when it belongs to the
        leaf's player and ``-value`` otherwise, then the running value is updated to
        ``node.reward + discount * value`` so that the parent sees the discounted return. Decision
        nodes carry the transition reward; chance nodes carry a zero reward.

        :param search_path: the nodes visited this simulation, from root to leaf
        :param value: the leaf bootstrap value, from ``to_play``'s perspective
        :param to_play: the player the leaf value is expressed from
        :param min_max_stats: the running value normalizer to update with each node's mean value
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())
            reward = node.reward if isinstance(node, DecisionNode) else 0.0
            value = reward + self.config.discount * value

    def _action_onehot(self, action: int) -> th.Tensor:
        """
        Build a batch-size-1 one-hot action tensor for the afterstate dynamics.

        :param action: the action id to encode
        :return: a tensor of shape ``[1, num_actions]`` with a single active entry, on the search device
        """
        onehot = th.zeros((1, self.config.num_actions), device=self._device)
        onehot[0, action] = 1.0
        return onehot

    def _chance_onehot(self, code: int) -> th.Tensor:
        """
        Build a batch-size-1 one-hot chance tensor for the dynamics.

        :param code: the codebook index to encode
        :return: a tensor of shape ``[1, codebook_size]`` with a single active entry, on the search device
        """
        onehot = th.zeros((1, self.config.codebook_size), device=self._device)
        onehot[0, code] = 1.0
        return onehot


# Raised when selection is attempted on a node that has not been expanded (has no children).
_UNEXPANDED_SELECTION_ERROR = "Cannot select a child of an unexpanded node."
