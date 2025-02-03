import pygame as pg

from rlgammon.environment.render_data.colors import Colors
from rlgammon.environment.render_data.board_parameters import BoardParameters


class Drawer:
    """
    TODO
    """

    def __init__(self, screen: pg.Surface):
        """
        TODO

        :param screen: xx
        """

        self.screen = screen

    def draw_playing_board(self):
        """
        Draw the board, on the provided display, where the backgammon game is played.
        """

        board_rect = pg.Rect(BoardParameters.margin, BoardParameters.margin,
                                 BoardParameters.board_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bg_color, board_rect)
        pg.draw.rect(self.screen, Colors.outline_color, board_rect, 3)

    def draw_central_bar(self, bar_x: int):
        """
        TODO

        :param bar_x: x coordinate (in pixels) of the top-left corner of the bar
        """

        bar_rect = pg.Rect(bar_x, BoardParameters.margin,
                               BoardParameters.bar_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bar_color, bar_rect)
        pg.draw.rect(self.screen, Colors.outline_color, bar_rect, 2)

    def draw_off_board_column(self):
        """
        TODO
        """

        off_rect = pg.Rect(BoardParameters.margin + BoardParameters.board_width,
                               BoardParameters.margin, BoardParameters.off_width, BoardParameters.board_height)
        pg.draw.rect(self.screen, Colors.bg_color, off_rect)
        pg.draw.rect(self.screen, Colors.outline_color, off_rect, 3)

    def draw_triangle(self):
        raise NotImplementedError
