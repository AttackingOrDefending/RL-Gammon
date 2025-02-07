"""Stores colors used in rendering the backgammon environment."""

from dataclasses import dataclass

from rlgammon.rlgammon_types import Color


@dataclass
class Colors:
    """
    Class for storing colors used in rendering the backgammon environment.

    All colors represented as RGB tuples.
    """

    bg_color: Color = (34, 139, 34)                                 # background color of the pygame screen (dark green)
    triangle_color1: Color = (222, 184, 135)                        # color of the first triangle set (light brown)
    triangle_color2: Color = (139, 69, 19)                          # color of the second triangle set (dark brown)
    triangle_colors: tuple[Color, Color] = (triangle_color1, triangle_color2)  # tuple with both triangle colors
    bar_color: Color = (105, 105, 105)                              # color of the middle dividing bar (gray)
    outline_color: Color = (0, 0, 0)                                # color of the board outline (black)
    player1_checker_color: Color = (255, 255, 255)                  # color of the start player checkers (black)
    player2_checker_color: Color = (0, 0, 0)                        # color of the second player checkers (white)
