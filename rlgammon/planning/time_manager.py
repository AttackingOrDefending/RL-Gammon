"""Allocate a per-move time budget (and a monotonic deadline) from a remaining clock.

All times are expressed in SECONDS. Deadlines are ``time.monotonic()`` timestamps, intended to be
fed straight into :meth:`rlgammon.planning.base_search.BaseSearch.search`'s ``deadline`` argument.
"""

import math
import time

# Default number of own moves assumed to remain at the start of a game.
DEFAULT_MOVES_TO_GO = 30
# Default lower bound (seconds) on any allocated per-move budget.
DEFAULT_MIN_MOVE_TIME = 0.05
# Default fraction of the remaining clock the manager is willing to spend overall.
DEFAULT_SAFETY_FRACTION = 0.9
# Default per-move overhead (seconds) reserved for move generation, I/O and clock latency.
DEFAULT_OVERHEAD = 0.02
# Smallest number of remaining moves the estimate is allowed to fall to (avoids spending it all at once).
MIN_ESTIMATED_REMAINING_MOVES = 1
# A full game ply is two half-moves (one per player), so ``move_number`` advances the estimate by halves.
HALF_MOVES_PER_FULL_MOVE = 2


class TimeManager:
    """Turn a remaining time budget into a per-move allocation and a monotonic deadline."""

    def __init__(self, *, moves_to_go: int = DEFAULT_MOVES_TO_GO, min_move_time: float = DEFAULT_MIN_MOVE_TIME,
                 safety_fraction: float = DEFAULT_SAFETY_FRACTION, overhead: float = DEFAULT_OVERHEAD) -> None:
        """
        Construct the time manager with the policy knobs that shape every allocation.

        :param moves_to_go: assumed number of own moves remaining at the start of the game
        :param min_move_time: lower bound in seconds on any allocated per-move budget
        :param safety_fraction: fraction of the remaining clock the manager is willing to spend overall
        :param overhead: per-move overhead in seconds reserved for generation, I/O and clock latency
        """
        self._moves_to_go = moves_to_go
        self._min_move_time = min_move_time
        self._safety_fraction = safety_fraction
        self._overhead = overhead

    def budget_for_move(self, time_left: float, move_number: int = 0) -> float | None:
        """
        Return the number of seconds to allocate to the move about to be played.

        The estimate of remaining moves shrinks as ``move_number`` grows, so the same clock is spread
        across fewer moves later in the game; the result is clamped to at least ``min_move_time`` and
        is never allowed to exceed ``time_left``.

        :param time_left: the remaining clock in seconds (``inf``/non-finite means an unlimited clock)
        :param move_number: the index of the move about to be played (counts both players' half-moves)
        :return: the seconds to allocate this move, or ``None`` if the clock is unlimited
        """
        if not math.isfinite(time_left):
            return None
        estimated_remaining = max(self._moves_to_go - move_number // HALF_MOVES_PER_FULL_MOVE,
                                  MIN_ESTIMATED_REMAINING_MOVES)
        budget = (time_left * self._safety_fraction) / estimated_remaining - self._overhead
        budget = max(budget, self._min_move_time)
        return min(budget, time_left)

    def deadline_for_move(self, time_left: float, move_number: int = 0,
                          now: float | None = None) -> float | None:
        """
        Return a monotonic deadline (``now + budget``) for the move about to be played.

        :param time_left: the remaining clock in seconds (``inf``/non-finite means an unlimited clock)
        :param move_number: the index of the move about to be played (counts both players' half-moves)
        :param now: the reference ``time.monotonic()`` timestamp (defaults to ``time.monotonic()``)
        :return: the monotonic deadline timestamp, or ``None`` if the clock is unlimited
        """
        budget = self.budget_for_move(time_left, move_number)
        if budget is None:
            return None
        reference = time.monotonic() if now is None else now
        return reference + budget
