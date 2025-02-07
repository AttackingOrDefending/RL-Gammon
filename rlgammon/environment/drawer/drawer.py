"""Draw different elements on a pygame screen to render the backgammon board in a human-friendly way."""

import pygame as pg

from rlgammon.environment.render_data.board_parameters import BoardParameters
from rlgammon.environment.render_data.colors import Colors
from rlgammon.rlgammon_types import Color


class Drawer:
    """Class for drawing different elements on a pygame screen to render the backgammon board in a human-friendly way."""

    def __init__(self, screen: pg.Surface) -> None:
        """
        Initialize the drawer with the given screen.

        :param screen: the pygame screen to draw on, which should be displayed during rendering
        """
        self.screen = screen

    def draw_playing_board(self) -> None:
        """Draw the board, on the provided display, where the backgammon game is played."""
        board_rect = pg.Rect(BoardParameters.margin, BoardParameters.margin,
                                 BoardParameters.board_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bg_color, board_rect)
        pg.draw.rect(self.screen, Colors.outline_color, board_rect, 3)

    def draw_central_bar(self, bar_x: int) -> None:
        """
        Draw the central bar splitting the board into the left and right sections.

        :param bar_x: x coordinate (in pixels) of the top-left corner of the bar
        """
        bar_rect = pg.Rect(bar_x, BoardParameters.margin,
                               BoardParameters.bar_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bar_color, bar_rect)
        pg.draw.rect(self.screen, Colors.outline_color, bar_rect, 2)

    def draw_off_board_column(self) -> None:
        """Draw the column where checkers are borne-off."""
        off_rect = pg.Rect(BoardParameters.margin + BoardParameters.board_width,
                               BoardParameters.margin, BoardParameters.off_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bg_color, off_rect)
        pg.draw.rect(self.screen, Colors.outline_color, off_rect, 3)

    def draw_triangle(self, triangle_color: Color, points: list[tuple[float, float]]) -> None:
        """
        Draw a single triangle, where checkers are stacked, with the provided points.

        :param triangle_color: color of the triangle
        :param points: 3 points defining the triangle on the pygame screen
        """
        pg.draw.polygon(self.screen, triangle_color, points)
        pg.draw.polygon(self.screen, Colors.outline_color, points, 1)

    def draw_checker(self, checker_color: Color, checker_position: tuple[float, float]) -> None:
        """
        Draw a single checker at the specified position.

        :param checker_color: color of the checker
        :param checker_position: position of the checker on the pygame screen
        """
        pg.draw.circle(self.screen, checker_color, checker_position, BoardParameters.checker_radius)
        pg.draw.circle(self.screen, Colors.outline_color, checker_position, BoardParameters.checker_radius, 1)
