"""Handle game mechanics for backgammon."""

from collections.abc import Iterable

import numpy as np

from rlgammon.rlgammon_types import MoveDict, MovePart

# Constant representing the bar location (when pieces are knocked out)
BAR_LOC = 24

TOTAL_PIECES = 15
QUARTER_BOARD_SIZE = 6


class Backgammon:
    """Handle game mechanics for backgammon.

    This class manages the state and rules of a backgammon game, including:
    - Board position tracking
    - Move validation
    - Game state management
    - Board visualization
    """

    def __init__(self) -> None:
        """Initialize the backgammon board."""
        # board: numpy array representing 24 points on the board (negative for black, positive for white)
        self.board = np.zeros(24, dtype=np.int8)
        # bar: array storing knocked out pieces [white_pieces, black_pieces]
        self.bar = np.zeros(2, dtype=np.int8)
        # off: array tracking pieces that have been borne off [white_pieces, black_pieces]
        self.off = np.zeros(2, dtype=np.int8)
        self.reset()

    def reset(self) -> None:
        """Reset the backgammon board to starting position."""
        # Initialize empty board
        self.board = np.zeros(24, dtype=np.int8)
        self.bar = np.zeros(2, dtype=np.int8)
        self.off = np.zeros(2, dtype=np.int8)
        # Set up initial position (negative numbers for black, positive for white)
        self.board[0] = -2  # Black has 2 pieces on point 24
        self.board[5] = 5  # White has 5 pieces on point 19
        self.board[7] = 3  # White has 3 pieces on point 17
        self.board[11] = -5  # Black has 5 pieces on point 13
        self.board[12] = 5  # White has 5 pieces on point 12
        self.board[16] = -3  # Black has 3 pieces on point 8
        self.board[18] = -5  # Black has 5 pieces on point 6
        self.board[23] = 2  # White has 2 pieces on point 1

    def flip(self) -> None:
        """Flip the board to switch perspective between players."""
        # Negate all values to switch colors and reverse the board array
        self.board = -self.board[::-1]  # type: ignore[assignment]
        # Reverse bar and off arrays
        self.bar = self.bar[::-1]  # type: ignore[assignment]
        self.off = self.off[::-1]  # type: ignore[assignment]

    def get_bar_moves(self, dice: Iterable[int]) -> MoveDict:
        """Return all legal moves for the current player from the bar.

        :param dice: Collection of dice values available for moves
        :return: Dictionary mapping dice values to sets of legal moves from bar
                Each move is represented as (from_position, to_position)
        """
        possible_moves: MoveDict = {roll: set() for roll in set(dice)}

        # Check each dice roll for possible moves from bar
        for roll in dice:
            # Can move to a point if it has 1 or fewer opponent pieces
            if self.board[24 - roll] >= -1:
                possible_moves[roll].add((24, int(24 - roll)))
        return possible_moves

    def get_legal_moves(self, dice: Iterable[int]) -> MoveDict:
        """Return all legal moves for the current player.

        :param dice: Collection of dice values available for moves
        :return: Dictionary mapping dice values to sets of legal moves
                Each move is represented as (from_position, to_position)
        """
        # If pieces are on the bar, they must be moved first
        if self.bar[0] > 0:
            return self.get_bar_moves(dice)

        unique_dice = set(dice)
        possible_moves: MoveDict = {roll: set() for roll in unique_dice}
        our_checkers = np.flatnonzero(self.board > 0)

        # Check normal moves for each dice roll
        for roll in unique_dice:
            for loc in our_checkers:
                if loc - roll >= 0 and self.board[loc - roll] >= -1:
                    possible_moves[roll].add((int(loc), int(loc) - roll))

        # Check bearing off moves if all pieces are in home board
        if our_checkers.size == 0 or our_checkers[-1] < QUARTER_BOARD_SIZE:
            for roll in unique_dice:
                for loc in our_checkers:
                    if loc - roll < 0:
                        possible_moves[roll].add((int(loc), -1))
        return possible_moves

    def make_move(self, move: MovePart) -> tuple[int, int]:
        """Make a move on the board.

        :param move: Tuple of (from_position, to_position)
        :return: Tuple of (captures, bore_off) indicating number of pieces captured
                and borne off in this move
        """
        captures = 0
        bore_off = 0

        # Handle bearing off
        if move[1] == -1:
            self.board[move[0]] -= 1
            bore_off = 1
            self.off[0] += 1
        else:
            # Handle regular moves and captures
            if move[0] == BAR_LOC:
                self.bar[0] -= 1
            else:
                self.board[move[0]] -= 1

            # Handle landing on opponent's blot
            if self.board[move[1]] == -1:
                self.board[move[1]] = 1
                self.bar[1] += 1
                captures = 1
            else:
                self.board[move[1]] += 1
        return captures, bore_off

    def is_terminal(self) -> bool:
        """Return whether the game is over.

        :return: True if game is finished, False otherwise
        """
        return bool(self.off[0] == TOTAL_PIECES or self.off[1] == TOTAL_PIECES)

    def get_winner(self) -> int:
        """Return the winner of the game.

        :return: 1 if white wins, 0 if black wins
        """
        if self.off[0] == TOTAL_PIECES:
            return 1
        return 0


if __name__ == "__main__":
    # Example board setup for testing
    bg = Backgammon()
    bg.board[3] = 8  # 8 white pieces on point 21
    bg.bar[0] = 9  # 9 white pieces on the bar
    bg.bar[1] = 3  # 3 black pieces on the bar
    bg.off[0] = 5  # 5 white pieces borne off
    bg.off[1] = 8  # 8 black pieces borne off
