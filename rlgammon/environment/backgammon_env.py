"""A gym environment for backgammon."""

from collections.abc import Iterable
import random
from typing import Any

import numpy as np

from rlgammon.environment import backgammon as bg, human_renderer
from rlgammon.environment.text_renderer import text_render
from rlgammon.rlgammon_types import Board, Input, MoveDict, MovePart


class BackgammonEnv:
    """
    A gym environment for backgammon.

    Variables:
    - backgammon: Instance of the Backgammon game
    - max_moves: Maximum number of moves allowed in a game
    - moves: Current number of moves made in the game
    """

    def __init__(self) -> None:
        """
        Initialize the environment.

        Variables:
        - backgammon: New Backgammon game instance
        - max_moves: Set to 500 moves
        - moves: Initialize move counter to 0
        """
        super().__init__()
        self.backgammon: bg.Backgammon = bg.Backgammon()
        self.max_moves: int = 500
        self.moves: int = 0

    def get_input(self) -> Input:
        """
        Return the input for the current player.

        :return: Array containing board state, bar, and off information

        Variables:
        - our_pieces: Array containing positions of current player's pieces
        - enemy_pieces: Array containing positions of opponent's pieces
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
        return res

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Board, dict[str, Any]]:
        """
        Reset the environment.

        :param seed: Random seed for reproducibility
        :param options: Additional options for reset (unused)
        :return: Tuple of initial observation and empty dict

        Variables:
        - seed: Optional random seed for game initialization
        - moves: Reset to 0 for new game
        """
        self.seed(seed)
        self.backgammon.reset()
        self.moves = 0
        self.max_moves = options.get("max_moves", 500) if options is not None else 500
        return self.get_input(), {}

    @staticmethod
    def roll_dice() -> list[int]:
        """
        Roll the dice.

        :return: List of dice values (doubled if same value rolled)

        Variables:
        - rolls: List containing two dice roll values
        """
        rolls: list[int] = [random.randint(1, 6), random.randint(1, 6)]
        if rolls[0] == rolls[1]:
            return rolls * 2
        return rolls

    def flip(self) -> Board:
        """
        Flip the board.

        :return: New board state after flipping
        """
        self.backgammon.flip()
        return self.get_input()

    def step(self, action: MovePart) -> tuple[float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        :param action: Tuple of (from_point, to_point) representing the move
        :return: Tuple containing (observation, reward, done, truncated, info)

        Variables:
        - moves: Counter for number of moves made
        - done: Boolean indicating if game is finished
        - reward: Float value (1.0 for win, -1.0 for loss, 0.0 otherwise)
        """
        self.moves += 1
        self.backgammon.make_move(action)
        done = self.backgammon.is_terminal()
        reward = 0.
        if done:
            reward = 1. if self.backgammon.get_winner() == 1 else -1.
        return reward, done, self.moves >= self.max_moves and not done, {}

    def render(self, mode: str = "human") -> None:
        """
        Render the environment.

        :param mode: Rendering mode ("human" for GUI, else text-based)

        Variables:
        - render: Instance of BackgammonRenderer for GUI mode
        """
        if mode == "human":
            render = human_renderer.BackgammonRenderer()
            render.render(self.backgammon.board, self.backgammon.bar, self.backgammon.off)
        else:
            print(text_render(self.backgammon))

    @staticmethod
    def seed(seed: int | None = None) -> None:
        """
        Set the seed for the environment.

        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.Generator(np.random.MT19937(seed))

    def get_legal_moves(self, dice: Iterable[int]) -> MoveDict:
        """
        Return the legal moves for the current player.

        :param dice: Iterable of dice values
        :return: Dictionary mapping starting positions to sets of possible moves
        """
        return self.backgammon.get_legal_moves(dice)
