"""Stochastic UCT Monte-Carlo tree search over decision and chance nodes (negamax backups)."""

import math
import time

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.base_search import BaseSearch
from rlgammon.planning.planning_errors.planning_errors import NoLegalActionsError, SearchDepthError
from rlgammon.planning.planning_types import Evaluator, SearchResult

# Minimum allowed search depth (in decision plies).
MIN_DEPTH = 1
# Default number of simulations to run per search.
DEFAULT_NUM_SIMULATIONS = 800
# Default UCT exploration constant.
DEFAULT_C_UCT = 1.4


class _Node:
    """A single node of the search tree, either a decision node or a chance node."""

    def __init__(self, state: GameState, depth: int) -> None:
        """
        Construct a tree node wrapping a cloned game state.

        :param state: the game state this node represents (kept as an owned clone)
        :param depth: the remaining search depth (in decision plies) at this node
        """
        self.state = state
        self.depth = depth
        self.is_chance = state.is_chance_node()
        self.is_terminal = state.is_terminal()
        self.visits = 0
        # All values are stored from the root mover's perspective (negamax bookkeeping).
        self.value_sum = 0.0
        self.children: dict[int, _Node] = {}


class StochasticMCTS(BaseSearch):
    """Stochastic UCT search: UCT over decision nodes, probability sampling over chance nodes."""

    def __init__(self, evaluator: Evaluator, max_depth: int, *,
                 num_simulations: int = DEFAULT_NUM_SIMULATIONS, c_uct: float = DEFAULT_C_UCT,
                 rng: np.random.Generator | None = None) -> None:
        """
        Construct the stochastic MCTS search.

        :param evaluator: the leaf evaluator scoring newly expanded frontier states
        :param max_depth: the maximum search depth in decision plies (must be >= 1)
        :param num_simulations: the number of simulations to run per search
        :param c_uct: the UCT exploration constant
        :param rng: the random number generator for chance sampling (defaults to a fresh generator)
        :raises SearchDepthError: if ``max_depth`` is less than 1
        """
        super().__init__(evaluator, max_depth)
        if max_depth < MIN_DEPTH:
            raise SearchDepthError
        self._num_simulations = num_simulations
        self._c_uct = c_uct
        self._rng = rng if rng is not None else np.random.default_rng()
        self._nodes_visited = 0
        self._root_visit_counts: dict[int, int] = {}

    def get_visit_counts(self) -> dict[int, int]:
        """
        Return the per-action visit counts at the root of the most recent search.

        :return: a mapping from root action id to its visit count
        """
        return dict(self._root_visit_counts)

    def search(self, state: GameState, deadline: float | None = None) -> SearchResult:
        """
        Run UCT simulations from the root and return the most-visited action with statistics.

        With no ``deadline`` a fixed ``num_simulations`` are run. With a ``deadline`` simulations are
        run in a loop until the deadline is reached (the ``num_simulations`` cap is ignored), always
        running at least one simulation so the root has visited children.

        :param state: the root game state (a decision node) to search from
        :param deadline: an optional ``time.monotonic()`` timestamp to stop by (``None`` = fixed count)
        :return: the search result with the most-visited action, the root mean value and stats
        :raises NoLegalActionsError: if the root is a non-terminal, non-chance node with no moves
        """
        self._nodes_visited = 0
        root_mover = state.current_player()
        if not state.legal_actions():
            raise NoLegalActionsError

        root = _Node(state.clone(), self._max_depth)
        self._nodes_visited += 1
        if deadline is None:
            for _ in range(self._num_simulations):
                self._simulate(root, root_mover)
        else:
            # Always run at least one simulation, then keep going while time remains.
            self._simulate(root, root_mover)
            while time.monotonic() < deadline:
                self._simulate(root, root_mover)

        self._root_visit_counts = {
            action: child.visits for action, child in root.children.items()
        }
        best_action = max(self._root_visit_counts, key=lambda a: self._root_visit_counts[a])
        value = root.value_sum / root.visits if root.visits > 0 else 0.0
        return SearchResult(best_action=best_action, value=value,
                            nodes_visited=self._nodes_visited, pv=[best_action])

    def _simulate(self, root: _Node, root_mover: int) -> float:
        """
        Run a single simulation from the root, expanding one leaf and backing up its value.

        :param root: the root node of the tree
        :param root_mover: the player to move at the root (the perspective for all stored values)
        :return: the simulation's leaf value from ``root_mover``'s perspective
        """
        path: list[_Node] = [root]
        node = root
        while True:
            if node.is_terminal:
                value = node.state.returns()[root_mover]
                break
            if node.depth <= 0:
                value = self._evaluator.evaluate(node.state, root_mover)
                break
            child = self._expand_or_select(node, root_mover)
            path.append(child)
            if child.visits == 0:
                # Newly created leaf: evaluate it and stop descending this simulation.
                value = (child.state.returns()[root_mover] if child.is_terminal
                         else self._evaluator.evaluate(child.state, root_mover))
                break
            node = child

        for path_node in path:
            path_node.visits += 1
            path_node.value_sum += value
        return value

    def _expand_or_select(self, node: _Node, root_mover: int) -> _Node:
        """
        Pick (creating if needed) a child of ``node`` via sampling (chance) or UCT (decision).

        :param node: the internal node to descend from
        :param root_mover: the player to move at the root (perspective of stored values)
        :return: the selected (possibly freshly created) child node
        :raises NoLegalActionsError: if a decision node has no legal actions
        """
        if node.is_chance:
            outcome = self._sample_chance(node.state)
            return self._get_or_create_child(node, outcome, same_player_depth=node.depth)

        legal = node.state.legal_actions()
        if not legal:
            raise NoLegalActionsError
        action = self._select_uct(node, root_mover, legal)
        # A player move consumes one decision ply; the resulting node is a chance node.
        return self._get_or_create_child(node, action, same_player_depth=node.depth)

    def _get_or_create_child(self, node: _Node, action: int, same_player_depth: int) -> _Node:
        """
        Return the existing child for ``action`` or create it by applying the action to a clone.

        Decision-to-chance transitions keep the depth; chance-to-decision transitions (the next
        node is the opponent's decision) decrement it by one decision ply.

        :param node: the parent node
        :param action: the action id (a move at a decision node, a dice outcome at a chance node)
        :param same_player_depth: the parent's remaining depth, used to derive the child's depth
        :return: the child node for ``action``
        """
        if action in node.children:
            return node.children[action]
        child_state = node.state.clone()
        child_state.apply_action(action)
        child_depth = same_player_depth if node.is_chance is False else same_player_depth - 1
        child = _Node(child_state, child_depth)
        node.children[action] = child
        self._nodes_visited += 1
        return child

    def _select_uct(self, node: _Node, root_mover: int, legal: list[int]) -> int:
        """
        Select an action at a decision node by the UCT rule, prioritizing unexplored children.

        The exploitation term uses the child mean value expressed from the perspective of the
        player to move at ``node``: stored values are root-mover-centric, so they are negated when
        the player to move is the opponent of the root mover.

        :param node: the decision node to select an action at
        :param root_mover: the player to move at the root (perspective of stored values)
        :param legal: the legal action ids at ``node``
        :return: the selected action id
        """
        mover = node.state.current_player()
        sign = 1.0 if mover == root_mover else -1.0
        log_parent = math.log(node.visits + 1)
        best_action = legal[0]
        best_score = -math.inf
        for action in legal:
            child = node.children.get(action)
            if child is None or child.visits == 0:
                # Unexplored children are always prioritized over explored ones.
                return action
            exploit = sign * (child.value_sum / child.visits)
            explore = self._c_uct * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _sample_chance(self, state: GameState) -> int:
        """
        Sample a dice outcome at a chance node according to its true probabilities.

        :param state: the chance-node game state
        :return: the sampled chance action id
        """
        outcomes = state.chance_outcomes()
        actions = [action for action, _ in outcomes]
        probs = [prob for _, prob in outcomes]
        return int(self._rng.choice(actions, p=probs))
