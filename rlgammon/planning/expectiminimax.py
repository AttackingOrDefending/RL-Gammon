"""Expectiminimax search with star1/star2 alpha-beta pruning over chance nodes (negamax form)."""

import time

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.base_search import BaseSearch
from rlgammon.planning.planning_errors.planning_errors import NoLegalActionsError, SearchDepthError
from rlgammon.planning.planning_types import Evaluator, SearchResult

# Minimum allowed search depth (in decision plies).
MIN_DEPTH = 1
# Iterative-deepening depth cap used when a deadline is given but ``max_depth`` is effectively unbounded.
ITERATIVE_DEEPENING_CAP = 64


class StarMinimax(BaseSearch):
    """Expectiminimax search using negamax with star1/star2 pruning across chance nodes."""

    def __init__(self, evaluator: Evaluator, max_depth: int, *, use_star2: bool = True,
                 value_bounds: tuple[float, float] = (-3.0, 3.0)) -> None:
        """
        Construct the star-minimax search.

        :param evaluator: the leaf evaluator scoring non-terminal frontier states
        :param max_depth: the maximum search depth in decision plies (must be >= 1)
        :param use_star2: whether to enable the star2 probing phase on chance nodes
        :param value_bounds: the (lower, upper) bounds on any leaf value, used by star1/star2
        :raises SearchDepthError: if ``max_depth`` is less than 1
        """
        super().__init__(evaluator, max_depth)
        if max_depth < MIN_DEPTH:
            raise SearchDepthError
        self._use_star2 = use_star2
        self._lower, self._upper = value_bounds
        self._nodes_visited = 0

    def search(self, state: GameState, deadline: float | None = None) -> SearchResult:
        """
        Search the root decision node and return the negamax-optimal action with statistics.

        With no ``deadline`` the root is searched once at ``max_depth``. With a ``deadline`` the search
        deepens iteratively (depth 1, 2, ... up to ``max_depth``), keeping the best fully-completed
        result and stopping before any iteration that would start past the deadline (an anytime
        guarantee): if even depth 1 cannot finish, the best action found so far is still returned.

        :param state: the root game state (a decision node) to search from
        :param deadline: an optional ``time.monotonic()`` timestamp to stop by (``None`` = fixed depth)
        :return: the search result with the best action, its value and the (accumulated) node count
        :raises NoLegalActionsError: if the root is a non-terminal, non-chance node with no moves
        """
        self._nodes_visited = 0
        mover = state.current_player()
        legal = state.legal_actions()
        if not legal:
            raise NoLegalActionsError

        if deadline is None:
            result, _ = self._search_root(state, mover, legal, self._max_depth, None)
            return result

        best = SearchResult(best_action=legal[0], value=self._lower - 1.0,
                            nodes_visited=0, pv=[legal[0]])
        max_iteration_depth = min(self._max_depth, ITERATIVE_DEEPENING_CAP)
        for depth in range(MIN_DEPTH, max_iteration_depth + 1):
            if time.monotonic() >= deadline:
                break
            result, completed = self._search_root(state, mover, legal, depth, deadline)
            if completed or depth == MIN_DEPTH:
                # A completed deeper iteration supersedes the previous; an unfinished depth 1 still
                # yields the best action found so far (graceful), but no deeper partial is kept.
                best = result
            if not completed:
                break
        return SearchResult(best_action=best.best_action, value=best.value,
                            nodes_visited=self._nodes_visited, pv=best.pv)

    def _search_root(self, state: GameState, mover: int, legal: list[int], depth: int,
                     deadline: float | None) -> tuple[SearchResult, bool]:
        """
        Run a single fixed-depth root search, optionally bailing out between children at a deadline.

        :param state: the root game state (a decision node) to search from
        :param mover: the player to move at the root
        :param legal: the legal action ids at the root
        :param depth: the search depth in decision plies for this iteration
        :param deadline: an optional ``time.monotonic()`` timestamp to stop by between root children
        :return: the (best-so-far) result for this depth and whether every root child was searched
        """
        best_action = legal[0]
        best_value = self._lower - 1.0
        alpha = self._lower
        for action in self._ordered_actions(state, mover, legal):
            if deadline is not None and time.monotonic() >= deadline:
                return (SearchResult(best_action=best_action, value=best_value,
                                     nodes_visited=self._nodes_visited, pv=[best_action]), False)
            child = state.clone()
            child.apply_action(action)
            # The child is a chance node for the opponent's roll; its expectation is from `mover`.
            value = self._chance_value(child, depth, mover, alpha, self._upper)
            if value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, value)
        return (SearchResult(best_action=best_action, value=best_value,
                             nodes_visited=self._nodes_visited, pv=[best_action]), True)

    def _ordered_actions(self, state: GameState, mover: int, legal: list[int]) -> list[int]:
        """
        Order actions by a cheap one-ply leaf score (best first) to improve pruning.

        :param state: the decision-node state being expanded
        :param mover: the player to move at ``state``
        :param legal: the legal action ids to order
        :return: the legal actions ordered best-first from ``mover``'s perspective
        """
        scored: list[tuple[float, int]] = []
        for action in legal:
            child = state.clone()
            child.apply_action(action)
            self._nodes_visited += 1
            # A terminal child is scored exactly; otherwise score the mover's own view as a proxy.
            score = (child.returns()[mover] if child.is_terminal()
                     else self._evaluator.evaluate(child, mover))
            scored.append((score, action))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [action for _, action in scored]

    def _decision_value(self, state: GameState, depth: int, mover: int,
                        alpha: float, beta: float) -> float:
        """
        Return the negamax value of a decision node from ``mover``'s perspective.

        :param state: the decision-node (or terminal) game state
        :param depth: the remaining search depth in decision plies
        :param mover: the player to move at ``state``
        :param alpha: the lower bound of the alpha-beta window
        :param beta: the upper bound of the alpha-beta window
        :return: ``mover``'s value of the state
        :raises NoLegalActionsError: if the node is non-terminal, non-chance and has no moves
        """
        self._nodes_visited += 1
        if state.is_terminal():
            return state.returns()[mover]
        if depth <= 0:
            return self._evaluator.evaluate(state, mover)

        legal = state.legal_actions()
        if not legal:
            raise NoLegalActionsError

        best = self._lower - 1.0
        for action in self._ordered_actions(state, mover, legal):
            child = state.clone()
            child.apply_action(action)
            value = self._chance_value(child, depth, mover, alpha, beta)
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    def _chance_value(self, state: GameState, depth: int, mover: int,
                      alpha: float, beta: float) -> float:
        """
        Return the probability-weighted negamax value of a chance node from ``mover``'s view.

        The chance node resolves the opponent's roll, after which it is the opponent's decision;
        each child decision value is negated to convert it back to ``mover``'s perspective. Star1
        (and optionally star2) pruning narrows each child's window so it can be skipped when the
        running expectation already falls outside ``[alpha, beta]``.

        :param state: the chance-node (or terminal) game state
        :param depth: the remaining search depth in decision plies
        :param mover: the player from whose perspective the value is returned
        :param alpha: the lower bound of the alpha-beta window
        :param beta: the upper bound of the alpha-beta window
        :return: ``mover``'s probability-weighted value of the chance node
        """
        self._nodes_visited += 1
        if state.is_terminal():
            return state.returns()[mover]
        if not state.is_chance_node():
            # Defensive: treat an unexpected decision node here as the opponent's move.
            return -self._decision_value(state, depth - 1, state.current_player(), -beta, -alpha)

        outcomes = state.chance_outcomes()
        if self._use_star2:
            probed = self._star2_probe(state, outcomes, depth, alpha)
            if probed is not None:
                return probed
        return self._star1(state, outcomes, depth, alpha, beta)

    def _child_value(self, state: GameState, outcome: int, depth: int,
                     alpha: float, beta: float) -> float:
        """
        Apply a chance outcome and return the negated opponent-decision value (mover's view).

        :param state: the chance-node game state
        :param outcome: the chance action id (dice outcome) to apply
        :param depth: the remaining search depth in decision plies
        :param alpha: the lower bound of the (mover-perspective) window for this child
        :param beta: the upper bound of the (mover-perspective) window for this child
        :return: the mover's value of the resulting opponent decision node
        """
        child = state.clone()
        child.apply_action(outcome)
        opponent = child.current_player()
        # Negamax: the opponent decision is searched with the negated, swapped window.
        return -self._decision_value(child, depth - 1, opponent, -beta, -alpha)

    def _star1(self, state: GameState, outcomes: list[tuple[int, float]], depth: int,
               alpha: float, beta: float) -> float:
        """
        Compute the chance-node expectation with star1 pruning.

        :param state: the chance-node game state
        :param outcomes: the (action id, probability) pairs of the chance event
        :param depth: the remaining search depth in decision plies
        :param alpha: the lower bound of the alpha-beta window
        :param beta: the upper bound of the alpha-beta window
        :return: the (possibly bounded) expectation of the chance node from the mover's view
        """
        probs = [prob for _, prob in outcomes]
        done = 0.0
        for i, (outcome, prob) in enumerate(outcomes):
            future = probs[i + 1:]
            lower_future = sum(p * self._lower for p in future)
            upper_future = sum(p * self._upper for p in future)
            # Child window so the running expectation can still reach (alpha, beta).
            child_alpha = max(self._lower, (alpha - done - upper_future) / prob)
            child_beta = min(self._upper, (beta - done - lower_future) / prob)
            if child_alpha >= child_beta:
                value = self._child_value(state, outcome, depth, child_alpha, child_alpha)
            else:
                value = self._child_value(state, outcome, depth, child_alpha, child_beta)
            if value <= child_alpha:
                return done + prob * value + upper_future
            if value >= child_beta:
                return done + prob * value + lower_future
            done += prob * value
        return done

    def _star2_probe(self, state: GameState, outcomes: list[tuple[int, float]], depth: int,
                     alpha: float) -> float | None:
        """
        Run an incremental star2 probing pass to attempt an early fail-low against ``alpha``.

        Children are probed one at a time (the first child first) with a cheap null window at the
        threshold beta that would, with every other child at the global upper bound, make the
        chance-node expectation exactly ``alpha``. A probe returning below its beta yields a tight
        upper bound on that child and lowers the running optimistic expectation; the chance node
        can only be proven irrelevant when that optimistic bound (an upper bound on the true value)
        drops to at most ``alpha``, in which case it is returned as a value-preserving fail-low. If
        a probe instead fails high (no upper bound tighter than the global one) or a child would
        need an impossibly low value, the probe is abandoned and ``None`` is returned so the caller
        runs the full star1 pass.

        :param state: the chance-node game state
        :param outcomes: the (action id, probability) pairs of the chance event
        :param depth: the remaining search depth in decision plies
        :param alpha: the lower bound of the alpha-beta window
        :return: a fail-low bound (at most ``alpha``) if the probe proves a cutoff, else ``None``
        """
        probs = [prob for _, prob in outcomes]
        if sum(probs) <= 0.0:
            return None
        # Optimistic expectation with every child pinned at the global upper bound.
        optimistic = sum(p * self._upper for p in probs)
        for outcome, prob in outcomes:
            if optimistic <= alpha:
                # The accumulated upper bound on the true value already proves a fail-low.
                return optimistic
            # Threshold beta for this child that would bring the optimistic total down to alpha.
            others_upper = optimistic - prob * self._upper
            child_beta = (alpha - others_upper) / prob
            if child_beta <= self._lower:
                # This child would need an impossibly low value to reach alpha; probing won't help.
                return None
            # Cheap null-window probe: a return below child_beta is a valid upper bound on the child.
            probe = self._child_value(state, outcome, depth, child_beta, child_beta)
            if probe >= child_beta:
                # The probe failed high: no upper bound tighter than the global one; bail out.
                return None
            optimistic = others_upper + prob * probe
        return optimistic if optimistic <= alpha else None
