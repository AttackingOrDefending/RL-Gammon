"""A gym environment for backgammon."""

from __future__ import annotations

from collections.abc import Iterable
import random
import sys
from typing import Any

import numpy as np

from rlgammon.environment import backgammon as bg, human_renderer
from rlgammon.environment.text_renderer import text_render
from rlgammon.environment.utils.normalize_input import cell_stats, normalize_input
from rlgammon.rlgammon_types import Input, MoveList, MovePart


class BackgammonEnv:
    """
    A gym environment for backgammon.

    Variables:
    - backgammon: Instance of the Backgammon game.
    - max_moves: Maximum number of moves allowed in a game.
    - moves: Current number of moves made in the game.
    - _cache: Internal cache to speed up get_all_complete_moves.
    """

    def __init__(self) -> None:
        """
        Initialize the environment.

        Variables:
        - backgammon: New Backgammon game instance.
        - max_moves: Set to 500 moves.
        - moves: Initialize move counter to 0.
        - current_player: the current player's orientation of board
        - _cache: Dictionary used for caching computed move combinations.
        """
        super().__init__()
        self.backgammon: bg.Backgammon = bg.Backgammon()
        self.max_moves: int = 500
        self.moves: int = 0
        self.current_player = 1
        self.observation_shape = self.get_input().shape
        self.action_shape = 8
        self._cache: dict[
            tuple[int, tuple[int, ...]] | tuple[int, int, int],
            list[tuple[BackgammonEnv, list[tuple[int, MovePart]]]],
        ] = {}

    def has_lost(self, player: int) -> bool:
        """
        Basic way to check if a player has lost, by comparing the provided player
        to the player who should have played after the last move (i.e. the losing player).
        Have to be careful to only use it after done was returned to true and reward = 1.

        :param player: the player to check
        :return: true, if the player has lost, false otherwise.
        """
        return self.moves % 2 != player % 2

    def get_input(self, get_normalized: bool = False) -> Input:
        """
        Return the input for the current player.

        :param get_normalized: Whether to normalize the input.
        :return: Array containing board state, bar, and off information.
        """
        board = self.backgammon.board  # shape (24,)
        # Pre-allocate the result array:
        # our_pieces (24) + enemy_pieces (24) + bar (2) + off (2) = 52 elements.
        res = np.empty(52, dtype=np.int8)

        # Use np.maximum to replace elementwise comparison and assignment (vectorized).
        np.maximum(board, 0, out=res[:24])  # our_pieces: all negative values become 0
        np.maximum(-board, 0, out=res[24:48])  # enemy_pieces: all negative values (of -board) become 0

        # Directly set bar and off.
        res[48:50] = self.backgammon.bar
        res[50:52] = self.backgammon.off
        if get_normalized:
            return normalize_input(res, cell_stats)
        return res

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Input, dict[str, Any]]:
        """
        Reset the environment.

        :param seed: Random seed for reproducibility.
        :param options: Additional options for reset (unused).
        :return: Tuple of initial observation and an empty dict.
        """
        self.seed(seed)
        self.backgammon.reset()
        self.moves = 0
        self.max_moves = options.get("max_moves", 500) if options is not None else 500
        # Clear the cache on reset.
        self._cache.clear()
        return self.get_input(), {}

    @staticmethod
    def roll_dice() -> list[int]:
        """
        Roll the dice.

        :return: List of dice values (doubled if same value rolled).
        """
        rolls: list[int] = [random.randint(1, 6), random.randint(1, 6)]
        if rolls[0] == rolls[1]:
            return rolls * 2
        return rolls

    def flip(self) -> Input:
        """
        Flip the board.

        :return: New board state after flipping.
        """
        self.backgammon.flip()
        self.current_player *= -1
        return self.get_input()

    def step(self, action: MovePart) -> tuple[float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        :param action: Tuple of (from_point, to_point) representing the move.
        :return: Tuple containing (reward, done, truncated, info).
        """
        self.moves += 1
        white_pieces = (np.sum(self.backgammon.board[self.backgammon.board > 0]) + self.backgammon.off[0]
                        + self.backgammon.bar[0])
        black_pieces = (np.sum(-self.backgammon.board[self.backgammon.board < 0]) + self.backgammon.off[1]
                        + self.backgammon.bar[1])
        self.backgammon.make_move(action)
        white_pieces_new = (np.sum(-self.backgammon.board[self.backgammon.board < 0]) + self.backgammon.off[1]
                            + self.backgammon.bar[1])
        black_pieces_new = (np.sum(self.backgammon.board[self.backgammon.board > 0]) + self.backgammon.off[0]
                            + self.backgammon.bar[0])
        if black_pieces != black_pieces_new or white_pieces != white_pieces_new:
            self.render("text")
            print(action)
            print("Black pieces:", black_pieces, "Black pieces new:", black_pieces_new)
            print("White pieces:", white_pieces, "White pieces new:", white_pieces_new)
            sys.exit()
        done = self.backgammon.is_terminal()
        reward = 0.0
        if done:
            reward = self.backgammon.get_winner()
        return reward, done, self.moves >= self.max_moves and not done, {}

    def render(self, mode: str = "human") -> None:
        """
        Render the environment.

        :param mode: Rendering mode ("human" for GUI, else text-based).
        """
        if mode == "human":
            render = human_renderer.BackgammonRenderer()
            render.render(self.backgammon.board, self.backgammon.bar, self.backgammon.off)
        else:
            print(self)

    @staticmethod
    def seed(seed: int | None = None) -> None:
        """
        Set the seed for the environment.

        :param seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.Generator(np.random.MT19937(seed))

    def get_legal_moves(self, dice: Iterable[int]) -> MoveList:
        """
        Return the legal moves for the current player.

        :param dice: Iterable of dice values.
        :return: List of legal moves as tuples of (roll, move).
        """
        actions_per_roll = self.backgammon.get_legal_moves(dice)
        actions = []
        for roll in dice:
            actions += [(roll, move) for move in actions_per_roll[roll]]
        return actions

    def get_all_complete_moves(
        self,
        dice: list[int],
    ) -> list[tuple[BackgammonEnv, list[tuple[int, MovePart]]]]:
        """
        Return all possible complete moves for the current player.
        Uses internal caching (self._cache) to avoid redundant computations.
        Special handling is provided for lists of dice that contain the same value (doubles).

        :param dice: List of dice values.
        :return: List of all possible complete moves.
        """
        if not dice:
            return []
        # Use a canonical key for caching.
        key = (hash(self), dice[0], len(dice)) if len(dice) > 0 and len(set(dice)) == 1 else (hash(self), tuple(dice))

        if key in self._cache:
            return self._cache[key]

        actions = self.get_legal_moves(dice)
        if not actions:
            result: list[tuple[BackgammonEnv, list[tuple[int, MovePart]]]] = []
            self._cache[key] = result
            return result

        moves: list[tuple[BackgammonEnv, list[tuple[int, MovePart]]]] = []
        for roll, action in actions:
            board_copy = self.copy()
            board_copy.step(action)
            next_dice = dice.copy()
            next_dice.remove(roll)
            next_moves = board_copy.get_all_complete_moves(next_dice) if next_dice else []
            if next_moves:
                moves += [(position, [(roll, action), *move]) for position, move in next_moves]
            else:
                moves += [(board_copy, [(roll, action)])]

        max_moves = max(len(move) for _, move in moves) if moves else 0
        # Filter out moves that are not the longest.
        moves = [move for move in moves if len(move[1]) == max_moves]
        # If not all rolls can be used, use the one with the largest roll.

        if len(dice) > 1 and dice[0] != dice[1] and max_moves == 1:
            max_roll = max(dice)
            moves_max_roll = [move for move in moves if move[1][0][0] == max_roll]
            if moves_max_roll:
                moves = moves_max_roll

        for board, _ in moves:
            board.flip()

        self._cache[key] = moves
        return moves

    def copy(self) -> BackgammonEnv:
        """
        Return a copy of the current environment.
        Also propagates the internal cache to the new copy.
        """
        env = BackgammonEnv()
        env.backgammon = self.backgammon.copy()
        env.max_moves = self.max_moves
        env.moves = self.moves
        # Propagate the cache reference so that all copies share the same cache.
        env._cache = self._cache
        return env

    def __hash__(self) -> int:
        """Return the hash of the input array."""
        return hash(tuple(self.get_input().tolist()))  # type: ignore[arg-type]

    def __eq__(self, other: BackgammonEnv) -> bool:  # type: ignore[override]
        """Return whether the input arrays are equal."""
        return np.array_equal(self.get_input(), other.get_input())

    def __repr__(self) -> str:
        """Represent the environment with text."""
        return text_render(self.backgammon)
