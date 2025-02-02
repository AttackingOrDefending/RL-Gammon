from dataclasses import dataclass


@dataclass
class Colors:
    """
    Class for storing colors used in rendering the backgammon environment
    All colors represented as RGB tuples.
    """

    bg_color: tuple = (34, 139, 34)                 # dark green background
    triangle_color1: tuple = (222, 184, 135)        # light brown
    triangle_color2: tuple = (139, 69, 19)          # dark brown
    bar_color: tuple = (105, 105, 105)              # gray for the bar
    white_color: tuple = (255, 255, 255)
    black_color: tuple = (0, 0, 0)
    outline_color: tuple = (0, 0, 0)                # black for the outline
