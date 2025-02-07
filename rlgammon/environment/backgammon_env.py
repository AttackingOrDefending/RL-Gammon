"""A gym environment for backgammon."""

from collections.abc import Iterable
import random
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from rlgammon.environment import backgammon as bg, human_renderer
from rlgammon.environment.text_renderer import text_render


class BackgammonEnv(gym.Env[npt.NDArray[np.int8], tuple[int, int]]):
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

    def get_input(self) -> npt.NDArray[np.int8]:
        """
        Return the input for the current player.

        :return: Array containing board state, bar, and off information

        Variables:
        - our_pieces: Array containing positions of current player's pieces
        - enemy_pieces: Array containing positions of opponent's pieces
        """
        our_pieces: npt.NDArray[np.int8] = self.backgammon.board.copy()
        our_pieces[our_pieces < 0] = 0
        enemy_pieces: npt.NDArray[np.int8] = -self.backgammon.board.copy()
        enemy_pieces[enemy_pieces < 0] = 0
        return np.concatenate([our_pieces, enemy_pieces, self.backgammon.bar, self.backgammon.off], dtype=np.int8)

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[npt.NDArray[np.int8], dict[str, Any]]:
        """
        Reset the environment.

        :param seed: Random seed for reproducibility
        :param options: Additional options for reset (unused)
        :return: Tuple of initial observation and empty dict

        Variables:
        - seed: Optional random seed for game initialization
        - moves: Reset to 0 for new game
        """
        if seed is not None:
            random.seed(seed)
            np.random.Generator(np.random.MT19937(seed))
        self.backgammon.reset()
        self.moves = 0
        return self.get_input(), {}

    def roll_dice(self) -> list[int]:
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

    def flip(self) -> npt.NDArray[np.int8]:
        """
        Flip the board.

        :return: New board state after flipping
        """
        self.backgammon.flip()
        return self.get_input()

    def step(self, action: tuple[int, int]) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
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
        return self.get_input(), reward, done, self.moves >= self.max_moves and not done, {}

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

    def seed(self, seed: int | None = None) -> None:
        """
        Set the seed for the environment.

        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.Generator(np.random.MT19937(seed))

    def get_legal_moves(self, dice: Iterable[int]) -> dict[int, set[tuple[int, int]]]:
        """
        Return the legal moves for the current player.

        :param dice: Iterable of dice values
        :return: Dictionary mapping starting positions to sets of possible moves
        """
        return self.backgammon.get_legal_moves(dice)
