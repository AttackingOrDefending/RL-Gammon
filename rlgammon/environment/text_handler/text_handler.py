import pygame as pg

from rlgammon.environment.render_data.colors import Colors


class TextHandler:
    """
    TODO
    """

    def __init__(self, screen: pg.Surface, font: pg.font.Font):
        """
        TODO

        :param screen:
        :param font:
        """

        self.screen = screen
        self.font = font

    def render_checker_text(self, text_position: tuple, text: str):
        """
        TODO

        :param text_position:
        :param text:
        """

        text = self.font.render(text, True, Colors.outline_color)
        text_rect = text.get_rect(center=text_position)
        self.screen.blit(text, text_rect)