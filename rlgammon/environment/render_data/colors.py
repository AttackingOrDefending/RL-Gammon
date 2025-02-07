"""Stores colors used in rendering the backgammon environment."""

from dataclasses import dataclass


@dataclass
class Colors:
    """
    Class for storing colors used in rendering the backgammon environment.

    All colors represented as RGB tuples.
    """

    bg_color: tuple[int, int, int] = (34, 139, 34)                     # background color of the pygame screen (dark green)
    triangle_color1: tuple[int, int, int] = (222, 184, 135)            # color of the first triangle set (light brown)
    triangle_color2: tuple[int, int, int] = (139, 69, 19)              # color of the second triangle set (dark brown)
    bar_color: tuple[int, int, int] = (105, 105, 105)                  # color of the middle dividing bar (gray)
    outline_color: tuple[int, int, int] = (0, 0, 0)                    # color of the board outline (black)
    player1_checker_color: tuple[int, int, int] = (255, 255, 255)      # color of the start player checkers (black)
    player2_checker_color: tuple[int, int, int] = (0, 0, 0)            # color of the second player checkers (white)
