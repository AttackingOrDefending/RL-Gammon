"""Class for handling text rendering in the game."""

import pygame as pg

from rlgammon.environment.render_data.colors import Colors


class TextHandler:
    """Class for handling text rendering in the game."""

    def __init__(self, screen: pg.Surface, font: pg.font.Font) -> None:
        """
        Initialize the TextHandler class.

        :param screen: The pygame screen to render the text on.
        :param font: The font to use for rendering the text.
        """
        self.screen = screen
        self.font = font

    def render_checker_text(self, text_position: tuple[int, int], text: str) -> None:
        """
        Render the text for the checkers.

        :param text_position: The position to render the text.
        :param text: The text to render.
        """
        text_surface = self.font.render(text, True, Colors.outline_color)
        text_rect = text_surface.get_rect(center=text_position)
        self.screen.blit(text_surface, text_rect)
