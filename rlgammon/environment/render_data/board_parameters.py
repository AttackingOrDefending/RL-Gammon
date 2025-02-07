"""Stores the parameters of the backgammon board for rendering purposes."""

from dataclasses import dataclass


@dataclass
class BoardParameters:
    """Class for storing parameters (e.g. dimension) of rendering the backgammon board."""

    screen_width: int = 900  # width (in pixels) of the displayed pygame screen
    screen_height: int = 600  # height (in pixels) of the displayed pygame screen
    margin: int = 50  # top and left margin size
    board_width: int = 700  # width of the playable board
    board_height: int = 500  # height of the playable board
    bar_width: int = 50  # width of the central, dividing bar
    off_width: int = 60  # off-board column width:

    triangle_height: int = (board_height // 2) - 20  # The triangle (point) areas:
    triangle_width = (board_width - bar_width) // 12  # There are 12 triangles per half.
