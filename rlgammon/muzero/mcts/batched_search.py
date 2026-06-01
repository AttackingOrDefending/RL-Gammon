"""Batched Gumbel Stochastic-MuZero search advancing many trees in lockstep on the GPU.

The single-tree :class:`~rlgammon.muzero.mcts.search.StochasticMuZeroMCTS` issues one batch-1 network
call per node expansion, which is launch-bound on a GPU. This module instead runs ``K`` independent
search trees SIMULTANEOUSLY: every simulation round collects one leaf from each active tree and
evaluates them all in a single batched network call (afterstate expansions in one call, decision
expansions in another), so the GPU sees batches of size up to ``K`` instead of one.

Root action selection follows Gumbel MuZero (Danihelka et al., 2022): the root is restricted to the
legal actions, ``m`` of them are sampled by Gumbel-top-k over the (legal) policy logits, the
simulation budget is spread over the sequential-halving phases (see
:func:`~rlgammon.muzero.mcts.gumbel.sequential_halving_schedule`), and the surviving set is halved by
the Gumbel-argmax score ``g(a) + logit(a) + sigma(completed_q(a))``. Non-root decision nodes and
chance nodes use the same pUCT rule as the single-tree search. The search returns, per tree, the
Gumbel-selected action and the Gumbel-improved policy used as the stored policy target.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import torch as th

from rlgammon.muzero.mcts.gumbel import (
    DEFAULT_C_SCALE,
    DEFAULT_C_VISIT,
    completed_q_values,
    gumbel_improved_policy,
    sample_gumbel,
    sequential_halving_schedule,
)
from rlgammon.muzero.mcts.node import ChanceNode, DecisionNode, MinMaxStats
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork


@dataclass
class GumbelRootResult:
    """The per-tree output of a batched Gumbel search: the chosen action and the policy target."""

    action: int
    policy: dict[int, float]
    root_value: float


@dataclass
class _Tree:
    """The mutable per-game search state advanced in lockstep by :class:`BatchedGumbelMCTS`."""

    root: DecisionNode
    legal_actions: list[int]
    min_max_stats: MinMaxStats
    # The actions still in contention under sequential halving (shrinks each phase).
    considered: list[int]
    # Gumbel(0) noise per legal action, indexed by action id (only legal entries are set).
    gumbel: dict[int, float]
    # The prior logit per legal action id (restricted-softmax input, pre-noise).
    logits: dict[int, float]
    root_value: float
    # Filled once the search finishes: the Gumbel-improved policy over the legal actions.
    result: GumbelRootResult | None = None
    # Scratch for the current simulation: the descent path and the pending leaf expansion request.
    search_path: list[DecisionNode | ChanceNode] = field(default_factory=list)
    pending_kind: int = 0
    pending_parent_state: th.Tensor | None = None
    pending_action: int = -1
    pending_chance_node: ChanceNode | None = None
    pending_afterstate: th.Tensor | None = None
    pending_code: int = -1
    pending_decision_node: DecisionNode | None = None
    pending_leaf_to_play: int = 0
    # The bootstrap value of the freshly expanded leaf, filled by the batched expansion call.
    pending_value: float = 0.0


# Leaf-expansion kinds collected during a simulation round (used to bucket the batched network calls).
_KIND_NONE = 0
_KIND_AFTERSTATE = 1
_KIND_DECISION = 2

# Default top-k prior fan-out of INTERNAL decision nodes (the root keeps all its legal actions). Over a
# 1352-action space, capping the internal fan-out is what keeps the Python per-node cost bounded.
DEFAULT_MAX_INTERNAL_CHILDREN = 16


class BatchedGumbelMCTS:
    """Run Gumbel Stochastic-MuZero search over many trees at once, batching every network call."""

    def __init__(self, config: MuZeroConfig, network: StochasticMuZeroNetwork,
                 rng: np.random.Generator, *, num_considered: int,
                 max_internal_children: int = DEFAULT_MAX_INTERNAL_CHILDREN) -> None:
        """
        Construct the batched search around a configuration, network, generator and root width.

        :param config: the configuration providing the pUCT, simulation and discount settings
        :param network: the learned Stochastic MuZero network providing every inference entry point
        :param rng: the random number generator for the per-root Gumbel noise
        :param num_considered: the number ``m`` of root actions Gumbel considers (capped at the legals)
        :param max_internal_children: the top-k prior fan-out cap of INTERNAL decision nodes; the search
            over a 1352-action space is dominated by per-node child bookkeeping, so internal nodes keep
            only their few highest-prior actions (the root is unaffected -- it keeps every legal action)
        """
        self.config = config
        self.network = network
        self.rng = rng
        self.num_considered = num_considered
        self.max_internal_children = max_internal_children
        self._device = th.device(config.device)

    def run_batch(self, root_observations: th.Tensor,
                  legal_actions_per_tree: list[list[int]]) -> list[GumbelRootResult]:
        """
        Run a full Gumbel search for every tree in lockstep and return their per-tree results.

        :param root_observations: the stacked root observations of shape ``[K, observation_size]``
        :param legal_actions_per_tree: the legal action ids of each of the ``K`` trees
        :return: the list of ``K`` :class:`GumbelRootResult`, one per input tree, in order
        """
        self.network.eval()
        trees = self._build_roots(root_observations, legal_actions_per_tree)
        schedule = sequential_halving_schedule(self.config.num_simulations, self.num_considered)

        for per_action in schedule:
            self._run_phase(trees, per_action)
            self._halve(trees)

        for tree in trees:
            self._finalize_tree(tree)
        return [tree.result for tree in trees if tree.result is not None]

    def _build_roots(self, root_observations: th.Tensor,
                     legal_actions_per_tree: list[list[int]]) -> list[_Tree]:
        """
        Run the batched initial inference and build one expanded, Gumbel-seeded root per tree.

        :param root_observations: the stacked root observations of shape ``[K, observation_size]``
        :param legal_actions_per_tree: the legal action ids of each tree
        :return: the list of initialised :class:`_Tree` states
        """
        observations = root_observations.to(self._device)
        with th.no_grad():
            output = self.network.initial_inference(observations)
            root_values = self.network.value_to_scalar(output.value)

        trees: list[_Tree] = []
        for index, legal_actions in enumerate(legal_actions_per_tree):
            policy_logits = output.policy_logits[index]
            root = DecisionNode(prior=0.0)
            root.expand(
                state=output.state[index], reward=0.0, to_play=0,
                policy_logits=policy_logits, legal_actions=legal_actions,
            )
            gumbel = sample_gumbel(len(legal_actions), self.rng)
            gumbel_by_action = {action: float(gumbel[i]) for i, action in enumerate(legal_actions)}
            logits_by_action = {action: float(policy_logits[action]) for action in legal_actions}
            considered = self._initial_considered(legal_actions, gumbel_by_action, logits_by_action)
            root_value = float(root_values[index])
            stats = MinMaxStats()
            stats.update(root_value)
            trees.append(_Tree(
                root=root, legal_actions=legal_actions, min_max_stats=stats,
                considered=considered, gumbel=gumbel_by_action, logits=logits_by_action,
                root_value=root_value,
            ))
        return trees

    def _initial_considered(self, legal_actions: list[int], gumbel: dict[int, float],
                            logits: dict[int, float]) -> list[int]:
        """
        Pick the initial ``m`` considered actions by Gumbel-top-k over ``g(a) + logit(a)``.

        :param legal_actions: the legal action ids of the tree
        :param gumbel: the Gumbel noise keyed by action id
        :param logits: the prior logit keyed by action id
        :return: the up-to-``m`` considered action ids, highest Gumbel-perturbed logit first
        """
        ranked = sorted(legal_actions, key=lambda action: gumbel[action] + logits[action], reverse=True)
        return ranked[: min(self.num_considered, len(ranked))]

    def _run_phase(self, trees: list[_Tree], per_action: int) -> None:
        """
        Run ``per_action`` simulations for every considered action of every tree, batched per round.

        Each round picks one (tree, considered action) leaf to expand from every active tree, then
        expands all collected leaves with at most two batched network calls (afterstate / decision).

        :param trees: the per-tree search states advanced in place
        :param per_action: the number of simulations each surviving action receives this phase
        """
        # The considered sets can differ in size across trees only at the final 1-action phase, so a
        # round index into a per-tree (action, repeat) list keeps every tree in lockstep.
        max_jobs = max((len(tree.considered) * per_action for tree in trees), default=0)
        for job in range(max_jobs):
            active = self._collect_round(trees, job, per_action)
            if active:
                self._expand_round(active)

    def _collect_round(self, trees: list[_Tree], job: int, per_action: int) -> list[_Tree]:
        """
        Descend one simulation in every tree whose round ``job`` is still within its workload.

        :param trees: the per-tree search states
        :param job: the round index (``action_slot * per_action + repeat``)
        :param per_action: the per-action simulation count of the phase
        :return: the trees that produced a leaf to expand this round
        """
        active: list[_Tree] = []
        for tree in trees:
            workload = len(tree.considered) * per_action
            if job >= workload:
                continue
            action_slot = job // per_action
            root_action = tree.considered[action_slot]
            if self._descend(tree, root_action):
                active.append(tree)
        return active

    def _descend(self, tree: _Tree, root_action: int) -> bool:
        """
        Descend from the root through a fixed first action to an unexpanded leaf, recording the path.

        The first edge is the Gumbel-chosen ``root_action``; deeper decision and chance nodes use the
        pUCT rule. The leaf-expansion request is stashed on the tree for the batched expansion step.

        :param tree: the tree to descend (its scratch fields are filled in place)
        :param root_action: the considered root action to simulate
        :return: ``True`` if a leaf to expand was found, ``False`` if the path hit only expanded nodes
        """
        root = tree.root
        chance_node = root.children[root_action]
        search_path: list[DecisionNode | ChanceNode] = [root, chance_node]
        if not chance_node.is_expanded():
            tree.pending_kind = _KIND_AFTERSTATE
            assert root.state is not None
            tree.pending_parent_state = root.state
            tree.pending_action = root_action
            tree.pending_chance_node = chance_node
            tree.pending_leaf_to_play = root.to_play
            tree.search_path = search_path
            return True

        code, next_decision = self._select_child_chance(chance_node, tree.min_max_stats)
        search_path.append(next_decision)
        return self._descend_from_decision(tree, next_decision, chance_node, code, search_path)

    def _descend_from_decision(self, tree: _Tree, decision_node: DecisionNode, parent_chance: ChanceNode,
                               code: int, search_path: list[DecisionNode | ChanceNode]) -> bool:
        """
        Continue a descent from a decision node reached after a chance roll, to an unexpanded leaf.

        :param tree: the tree being descended (scratch fields filled in place)
        :param decision_node: the decision node the descent currently sits at
        :param parent_chance: the chance node whose selected outcome produced ``decision_node``
        :param code: the codebook index selected at ``parent_chance``
        :param search_path: the path accumulated so far (root .. decision_node)
        :return: ``True`` once a leaf-expansion request has been stashed on the tree
        """
        if not decision_node.is_expanded():
            tree.pending_kind = _KIND_DECISION
            assert parent_chance.afterstate is not None
            tree.pending_afterstate = parent_chance.afterstate
            tree.pending_code = code
            tree.pending_decision_node = decision_node
            tree.pending_leaf_to_play = 1 - parent_chance.to_play
            tree.search_path = search_path
            return True

        while True:
            action, chance_node = self._select_child_decision(decision_node, tree.min_max_stats)
            search_path.append(chance_node)
            if not chance_node.is_expanded():
                tree.pending_kind = _KIND_AFTERSTATE
                assert decision_node.state is not None
                tree.pending_parent_state = decision_node.state
                tree.pending_action = action
                tree.pending_chance_node = chance_node
                tree.pending_leaf_to_play = decision_node.to_play
                tree.search_path = search_path
                return True

            code, next_decision = self._select_child_chance(chance_node, tree.min_max_stats)
            search_path.append(next_decision)
            if not next_decision.is_expanded():
                tree.pending_kind = _KIND_DECISION
                assert chance_node.afterstate is not None
                tree.pending_afterstate = chance_node.afterstate
                tree.pending_code = code
                tree.pending_decision_node = next_decision
                tree.pending_leaf_to_play = 1 - chance_node.to_play
                tree.search_path = search_path
                return True
            decision_node = next_decision

    def _expand_round(self, active: list[_Tree]) -> None:
        """
        Expand every active tree's pending leaf with at most two batched network calls, then back up.

        Afterstate-expansions are gathered into one :meth:`recurrent_inference_afterstate` call and
        decision-expansions into one :meth:`recurrent_inference_state` call, so a whole round of ``K``
        leaves costs two network launches instead of ``2K``.

        :param active: the trees that produced a leaf this round
        """
        afterstate_trees = [tree for tree in active if tree.pending_kind == _KIND_AFTERSTATE]
        decision_trees = [tree for tree in active if tree.pending_kind == _KIND_DECISION]

        if afterstate_trees:
            self._expand_afterstate_batch(afterstate_trees)
        if decision_trees:
            self._expand_decision_batch(decision_trees)

        for tree in active:
            self._backpropagate(tree)
            tree.pending_kind = _KIND_NONE

    def _expand_afterstate_batch(self, trees: list[_Tree]) -> None:
        """
        Batch-expand a list of leaf chance nodes via one afterstate-dynamics network call.

        :param trees: the trees whose pending leaf is an afterstate (chance-node) expansion
        """
        states = th.stack([tree.pending_parent_state for tree in trees
                           if tree.pending_parent_state is not None])
        actions = th.zeros((len(trees), self.config.num_actions), device=self._device)
        for row, tree in enumerate(trees):
            actions[row, tree.pending_action] = 1.0
        with th.no_grad():
            output = self.network.recurrent_inference_afterstate(states, actions)
            sigmas = th.softmax(output.chance_logits, dim=1)
            values = self.network.value_to_scalar(output.afterstate_value)
        for row, tree in enumerate(trees):
            chance_node = tree.pending_chance_node
            assert chance_node is not None
            chance_node.expand(afterstate=output.afterstate[row], to_play=tree.pending_leaf_to_play,
                               sigma=sigmas[row])
            tree.pending_value = float(values[row])

    def _expand_decision_batch(self, trees: list[_Tree]) -> None:
        """
        Batch-expand a list of leaf decision nodes via one dynamics network call.

        :param trees: the trees whose pending leaf is a decision-node expansion
        """
        afterstates = th.stack([tree.pending_afterstate for tree in trees
                                if tree.pending_afterstate is not None])
        codes = th.zeros((len(trees), self.config.codebook_size), device=self._device)
        for row, tree in enumerate(trees):
            codes[row, tree.pending_code] = 1.0
        with th.no_grad():
            output = self.network.recurrent_inference_state(afterstates, codes)
            rewards = self.network.reward_to_scalar(output.reward)
            values = self.network.value_to_scalar(output.value)
        for row, tree in enumerate(trees):
            decision_node = tree.pending_decision_node
            assert decision_node is not None
            decision_node.expand(state=output.state[row], reward=float(rewards[row]),
                                 to_play=tree.pending_leaf_to_play, policy_logits=output.policy_logits[row],
                                 legal_actions=None, max_children=self.max_internal_children)
            tree.pending_value = float(values[row])

    def _backpropagate(self, tree: _Tree) -> None:
        """
        Back the freshly expanded leaf value up the tree's recorded search path.

        The leaf value (from ``pending_leaf_to_play``'s perspective) is read from the tree's
        ``pending_value`` set by the batched expansion, so the per-tree backup is identical to the
        single-tree search: ``+value`` for nodes of the leaf's player and ``-value`` otherwise,
        folding each decision node's reward as it ascends.

        :param tree: the tree to back up
        """
        value = tree.pending_value
        to_play = tree.pending_leaf_to_play
        for node in reversed(tree.search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            tree.min_max_stats.update(node.value())
            reward = node.reward if isinstance(node, DecisionNode) else 0.0
            value = reward + self.config.discount * value

    def _halve(self, trees: list[_Tree]) -> None:
        """
        Halve every tree's considered set by the Gumbel-argmax score, keeping the better half.

        :param trees: the trees whose considered sets are halved in place
        """
        for tree in trees:
            if len(tree.considered) <= 1:
                continue
            keep = max(1, len(tree.considered) // 2)
            scored = sorted(
                tree.considered, key=lambda action: self._gumbel_score(tree, action), reverse=True,
            )
            tree.considered = scored[:keep]

    def _gumbel_score(self, tree: _Tree, action: int) -> float:
        """
        Compute the Gumbel-argmax selection score ``g(a) + logit(a) + sigma(q(a))`` for a root action.

        :param tree: the tree the action belongs to
        :param action: the root action id to score
        :return: the Gumbel selection score used for sequential halving and the final argmax
        """
        child = tree.root.children[action]
        q_value = self._normalized_child_q(tree, child)
        max_visit = self._max_child_visit(tree)
        # ``sigma`` is linear in q, so evaluate it as a scalar here to avoid per-call tensor creation
        # in this hot selection path (it matches :func:`~rlgammon.muzero.mcts.gumbel.sigma` exactly).
        scaled = (DEFAULT_C_VISIT + float(max_visit)) * DEFAULT_C_SCALE * q_value
        return tree.gumbel[action] + tree.logits[action] + scaled

    def _normalized_child_q(self, tree: _Tree, child: ChanceNode) -> float:
        """
        Return the root child's value from the root's perspective, normalized to ``[0, 1]``.

        A chance child of the root keeps the root's player, so no sign flip is needed; unvisited
        children fall back to the root value approximation.

        :param tree: the tree the child belongs to
        :param child: the root's chance-node child to score
        :return: the normalized completed Q value of the child
        """
        if child.visit_count == 0:
            return tree.min_max_stats.normalize(tree.root_value)
        return tree.min_max_stats.normalize(child.value())

    @staticmethod
    def _max_child_visit(tree: _Tree) -> int:
        """
        Return the maximum visit count over the root's children.

        :param tree: the tree to inspect
        :return: the largest child visit count (0 if the root has no visited children)
        """
        return max((child.visit_count for child in tree.root.children.values()), default=0)

    def _finalize_tree(self, tree: _Tree) -> None:
        """
        Select the Gumbel-argmax action and compute the Gumbel-improved policy target for one tree.

        The policy target is ``softmax(logit(a) + sigma(completed_q(a)))`` over the LEGAL actions, with
        unvisited actions completed by the root value approximation, matching Gumbel MuZero.

        :param tree: the tree to finalise (its ``result`` is set in place)
        """
        legal = tree.legal_actions
        logits = th.tensor([tree.logits[action] for action in legal])
        raw_q = th.zeros(len(legal))
        visits = th.zeros(len(legal))
        for i, action in enumerate(legal):
            child = tree.root.children[action]
            visits[i] = child.visit_count
            if child.visit_count > 0:
                raw_q[i] = tree.min_max_stats.normalize(child.value())
        value_prior = tree.min_max_stats.normalize(tree.root_value)
        completed = completed_q_values(th.softmax(logits, dim=0), raw_q, visits, value_prior)
        max_visit = self._max_child_visit(tree)
        improved = gumbel_improved_policy(logits, completed, max_visit)
        policy = {action: float(improved[i]) for i, action in enumerate(legal)}

        best_action = max(tree.considered, key=lambda action: self._gumbel_score(tree, action))
        tree.result = GumbelRootResult(action=best_action, policy=policy, root_value=tree.root_value)

    def _select_child_decision(self, node: DecisionNode,
                               min_max_stats: MinMaxStats) -> tuple[int, ChanceNode]:
        """
        Select an action at an internal decision node by the pUCT rule (mirrors the single-tree search).

        :param node: the decision node to select an action at
        :param min_max_stats: the running value normalizer for the exploitation term
        :return: a tuple ``(action, child_chance_node)`` of the selected action and its child
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
        assert best_child is not None
        return best_action, best_child

    def _select_child_chance(self, node: ChanceNode,
                             min_max_stats: MinMaxStats) -> tuple[int, DecisionNode]:
        """
        Select a chance outcome at a chance node by the pUCT rule using ``sigma`` as the prior.

        :param node: the chance node to select an outcome at
        :param min_max_stats: the running value normalizer for the exploitation term
        :return: a tuple ``(code, child_decision_node)`` of the selected codebook index and its child
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
        assert best_child is not None
        return best_code, best_child

    def _puct_score(self, child: DecisionNode | ChanceNode, c_puct: float, sqrt_total: float,
                    parent_to_play: int, min_max_stats: MinMaxStats) -> float:
        """
        Compute the pUCT score of a child from the parent's perspective (mirrors the single-tree search).

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
        Compute the visit-count-dependent pUCT coefficient (mirrors the single-tree search).

        :param parent_visit_count: the parent node's visit count
        :return: ``c_puct_init + log((visits + c_puct_base + 1) / c_puct_base)``
        """
        return self.config.c_puct_init + math.log(
            (parent_visit_count + self.config.c_puct_base + 1) / self.config.c_puct_base,
        )
