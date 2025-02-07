"""Stores the parameters of the backgammon board for rendering purposes."""

from dataclasses import dataclass


@dataclass
class BoardParameters:
    """Class for storing parameters (e.g. dimension) of rendering the backgammon board."""

    board_position_count: int = 24                                  # positions on the main board (with triangles)
    bar_position_count: int = 2                                     # positions in the bar section of the board
    off_position_count: int = 2                                     # positions in the borne off section of the board
    triangle_counts: int = board_position_count                     # number of triangles (both sides combined)
    triangle_count_per_side: int = triangle_counts // 2             # number of triangles on one side of the board
    screen_width: int = 900                                         # width (in pixels) of the displayed pygame screen
    screen_height: int = 600                                        # height (in pixels) of the displayed pygame screen
    margin: int = 50                                                # top and left margin size
    board_width: int = 700                                          # width of the playable board
    board_height: int = 500                                         # height of the playable board
    bar_width: int = 50                                             # width of the central, dividing bar
    off_width: int = 60                                             # off-board column width:
    triangle_height: float = (board_height / 2) - 20                # The triangle (point) areas:
    triangle_width: float = (board_width - bar_width) / 12          # There are 12 triangles per half.
    checker_radius: int = int(triangle_width * 0.4)                 # radius of the backgammon checker
    checker_diameter: int = int(2 * checker_radius)                 # diameter of the backgammon checker
    top_base_offset: float = checker_radius / 2                     # offset the checker from the base of the top triangles
    bottom_base_offset: float = checker_radius / 2                  # offset the checker from the base of the bottom triangles
    top_available: float = triangle_height - top_base_offset        # available vertical space in top triangles
    bottom_available: float = triangle_height - bottom_base_offset  # available vertical space in the bottom triangles
    spacing: int = 1                                                # spacing between each checker in a stack
