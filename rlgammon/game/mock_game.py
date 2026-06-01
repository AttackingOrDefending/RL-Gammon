"""A tiny, pure-Python race game implementing the engine protocols, for tests without OpenSpiel.

This is NOT backgammon; it is the smallest game that exercises every protocol method (chance
nodes with a non-uniform distribution, decision nodes with several legal actions, terminal
signed returns, perspective-dependent observations and deep cloning). It lets search, MuZero
and agent code be unit-tested with no `pyspiel` dependency and a known optimal answer.
"""

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.feature_extractor import N_OBS
from rlgammon.game.game_errors.game_errors import TerminalStateError
from rlgammon.rlgammon_types import BLACK, WHITE

# OpenSpiel-style sentinels for current_player() at chance/terminal nodes.
CHANCE_PLAYER = -1
TERMINAL_PLAYER = -4

# Game constants: first token to reach GOAL wins; a roll yields a step of 1, 2 or 3.
GOAL = 6
MAX_STEP = 3
# Unified action id space: ids 0..7 (only 1..3 are ever legal/sampled), matching the pyspiel style.
NUM_MOCK_ACTIONS = 8
# A deliberately non-uniform dice distribution so chance-sampling tests are meaningful.
CHANCE_DISTRIBUTION: list[tuple[int, float]] = [(1, 0.5), (2, 0.3), (3, 0.2)]


class MockState:
    """A single state of the mock race game; satisfies the GameState protocol structurally."""

    def __init__(self) -> None:
        """Construct the initial state: a chance node for WHITE's opening roll."""
        self._to_move: int = WHITE
        self._is_chance: bool = True
        self._terminal: bool = False
        self._dice: int = 0
        self._progress: list[int] = [0, 0]
        self._returns: list[float] = [0.0, 0.0]

    def current_player(self) -> int:
        """Return the player to move, or a chance/terminal sentinel."""
        if self._terminal:
            return TERMINAL_PLAYER
        if self._is_chance:
            return CHANCE_PLAYER
        return self._to_move

    def is_chance_node(self) -> bool:
        """Return whether a dice roll is pending."""
        return self._is_chance and not self._terminal

    def is_terminal(self) -> bool:
        """Return whether the game is over."""
        return self._terminal

    def legal_actions(self) -> list[int]:
        """Return the legal step sizes (1..dice) at a decision node, else an empty list."""
        if self._terminal or self._is_chance:
            return []
        return list(range(1, self._dice + 1))

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return the fixed dice distribution at a chance node, else an empty list."""
        if not self.is_chance_node():
            return []
        return list(CHANCE_DISTRIBUTION)

    def apply_action(self, action: int) -> None:
        """
        Apply a dice outcome (at a chance node) or a step (at a decision node) in place.

        :param action: the dice value at a chance node, or the step size at a decision node
        :raises TerminalStateError: if the state is already terminal
        """
        if self._terminal:
            raise TerminalStateError
        if self._is_chance:
            self._dice = action
            self._is_chance = False
            return
        self._progress[self._to_move] += action
        if self._progress[self._to_move] >= GOAL:
            self._terminal = True
            self._returns = [1.0, -1.0] if self._to_move == WHITE else [-1.0, 1.0]
        else:
            self._to_move = BLACK if self._to_move == WHITE else WHITE
            self._is_chance = True

    def observation_tensor(self, player: int) -> list[float]:
        """
        Return a length-200 perspective-dependent observation tensor.

        :param player: the player whose perspective to encode
        :return: the observation tensor (own progress, opponent progress, dice, chance flag, padding)
        """
        opponent = BLACK if player == WHITE else WHITE
        obs = [0.0] * N_OBS
        obs[0] = self._progress[player] / GOAL
        obs[1] = self._progress[opponent] / GOAL
        obs[2] = self._dice / MAX_STEP
        obs[3] = 1.0 if self._is_chance else 0.0
        return obs

    def returns(self) -> list[float]:
        """Return a copy of the per-player signed returns."""
        return list(self._returns)

    def clone(self) -> "MockState":
        """Return an independent deep copy of the state."""
        clone = MockState()
        clone._to_move = self._to_move
        clone._is_chance = self._is_chance
        clone._terminal = self._terminal
        clone._dice = self._dice
        clone._progress = list(self._progress)
        clone._returns = list(self._returns)
        return clone


class MockGame:
    """A factory for the mock race game; satisfies the BackgammonGame protocol structurally."""

    def new_initial_state(self) -> GameState:
        """Return a fresh initial state (a chance node for WHITE's opening roll)."""
        return MockState()

    def num_distinct_actions(self) -> int:
        """Return the size of the unified action id space."""
        return NUM_MOCK_ACTIONS

    @staticmethod
    def contrived_win_in_one(player: int) -> MockState:
        """
        Build a decision-node state where exactly one legal action wins immediately for ``player``.

        With progress 3/6 and a dice of 3, the legal steps are 1, 2 and 3; only the step of 3
        reaches the goal, so search/agent code has a single known-optimal action.

        :param player: the player to move (WHITE=0, BLACK=1)
        :return: the contrived decision-node state
        """
        state = MockState()
        state._to_move = player
        state._is_chance = False
        state._terminal = False
        state._dice = MAX_STEP
        win_threshold = GOAL - MAX_STEP
        state._progress = [win_threshold, win_threshold]
        return state

    @staticmethod
    def winning_action() -> int:
        """Return the unique winning action id for a :meth:`contrived_win_in_one` state."""
        return MAX_STEP
