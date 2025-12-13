"""TODO."""
import math
import time

import pyspiel

from rlgammon.rlgammon_types import WHITE


class SearchGNU:
    """TODO."""

    def __init__(self, agent):
        """TODO."""
        self.agent = agent
        self.memoization: dict[tuple[str, int], float] = {}

    def expectimax_value( self, state: pyspiel.BackgammonState, depth: int,
                          start_time: float, time_limit_sec: float) -> float:
        """
        Variable-depth expectimax for PySpiel Backgammon.
        There are NO chance nodes because dice are baked into legal actions.

        depth = 1 → evaluate leaf
        depth = 2 → player → opponent → eval
        depth = 3 → player → opponent → player → eval
        depth = N → alternating decision nodes until leaf

        Returns: float (value for the root player)
        """
        if state.is_terminal():
            return state.returns()[WHITE]

        if depth == 1 or time.time() - start_time >= time_limit_sec:
            features = state.observation_tensor(WHITE)[:198]
            return self.agent.evaluate_position(features).detach().numpy()

        # ---------- memoization ----------
        key = (state.history_str(), depth)
        if key in self.memoization:
            return self.memoization[key]

        legal_actions = state.legal_actions()
        if not legal_actions:
            features = state.observation_tensor(WHITE)[:198]
            value = self.agent.evaluate_position(features).detach().numpy()
            self.memoization[key] = value
            return value

        best = float("-inf")
        for action in state.legal_actions():
            child = state.child(action)
            val = self.expectimax_value(child, depth - 1, start_time, time_limit_sec)
            best = max(best, val)

            if time.time() - start_time >= time_limit_sec:
                break

        self.memoization[key] = best
        return best

    # ----------------------------------------------------------------------
    # Best action wrapper
    # ----------------------------------------------------------------------
    def best_action(self, state: pyspiel.BackgammonState, depth: int,
                    start_time: float, time_limit_sec: float) -> int | None:
        """Return the best root action using variable-depth expectimax."""
        legal = state.legal_actions()
        if not legal:
            return None

        best_val = -math.inf
        best_action_int = None

        for action in legal:
            child = state.child(action)
            value = self.expectimax_value(child, depth - 1, start_time, time_limit_sec)

            if value > best_val:
                best_val = value
                best_action_int = action

        return best_action_int
