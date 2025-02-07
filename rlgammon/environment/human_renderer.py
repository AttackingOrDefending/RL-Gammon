import pygame as pg
import sys
import math
import time

from rlgammon.environment.render_data.board_parameters import BoardParameters
from rlgammon.environment.render_data.colors import Colors
from rlgammon.environment.drawer.drawer import Drawer
from rlgammon.environment.text_handler.text_handler import TextHandler
from rlgammon.environment.helpers.orientations import Orientations


class BackgammonRenderer:
    """
    Class used to render the backgammon board in a human friendly way using pygame.
    """

    def __init__(self):
        """
        Initialize the Renderer by setting up the pygame screen and initializing helper classes.
        """

        pg.init()
        self.screen = pg.display.set_mode((BoardParameters.screen_width, BoardParameters.screen_height))
        pg.display.set_caption("Backgammon")

        # Prepare a font for drawing numbers on overloaded stacks.
        self.text_handler = TextHandler(self.screen, pg.font.SysFont(None, 20))
        
        # Init drawer class used for drawing objects on the pygame display
        self.drawer = Drawer(self.screen)

    def render(self, positions: list, bar: list, off: list, render_duration_in_s: float = 2.0):
        """
        Main render method, displaying the entire board with the provided state, for the specified amount of time.

        :param positions: a list with the checkers in the 24 board positions
        :param bar: a list with the amount of checkers of either players in the bar section of the board
        :param off: a list with the amount of checkers of either players in the borne-off section of the board
        :param render_duration_in_s: the duration for which to render the board in seconds
        """

        # Check if the game state is valid
        self._is_valid_input(positions, bar, off)

        # Clear the screen.
        self.screen.fill(Colors.bg_color)

        # Draw the board playing area.
        self.drawer.draw_playing_board()

        # Draw the central bar.
        bar_x = BoardParameters.margin + (BoardParameters.board_width - BoardParameters.bar_width) // 2
        self.drawer.draw_central_bar(bar_x)

        # Draw the off-board column to the right of the board.
        self.drawer.draw_off_board_column()

        # Top half: points 13-24 (drawn left-to-right)
        self._render_top_triangles()

        # Bottom half: points 1-12 (drawn right-to-left in each quadrant so they mirror the top)
        self._render_bottom_triangles()

        # Top half: points 13-24 (indices 12 to 23)
        self._render_checkers_in_top_triangles(positions)

        # Bottom half: points 1-12 (indices 0 to 11)
        self._render_checkers_in_bottom_triangles(positions)

        # Draw checkers on the bar
        self._render_checkers_in_bar(bar)

        # Draw the checkers that have been bear-off
        self._render_borne_off_checkers(off)

        # Start rendering the board
        start_time = time.time()
        while time.time() - start_time < render_duration_in_s:
            pg.display.flip()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.KEYDOWN:
                    return

    @staticmethod
    def _is_valid_input(positions: list, off: list, bar: list):
        """
        Check if the input provided to the 'render' method is valid.

        :param positions: a list with the checkers in the 24 board positions
        :param bar: a list with the amount of checkers of either players in the bar section of the board
        :param off: a list with the amount of checkers of either players in the borne-off section of the board
        """

        if len(positions) != BoardParameters.board_position_count:
            raise ValueError("Positions must be a list of length 24")
        if len(bar) != BoardParameters.bar_position_count:
            raise ValueError("Bar must be a list of length 2")
        if len(off) != BoardParameters.off_position_count:
            raise ValueError("Off must be a list of length 2")

    def _render_top_triangles(self):
        """
        Draw the 12 triangles at the top of the board (points 13 to 24) with alternating colors.
        """

        for i in range(BoardParameters.triangle_count_per_side):
            if i < BoardParameters.triangle_count_per_side // 2:
                # Left quadrant (points 13-18)
                x = BoardParameters.margin + i * BoardParameters.triangle_width
            else:
                # Right quadrant (points 19-24)
                x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                     BoardParameters.bar_width + (i - 6) * BoardParameters.triangle_width)
            y = BoardParameters.margin
            self.drawer.draw_triangle(Colors.triangle_colors[i % 2],
                                      self._get_points_from_coordinates(x, y, is_bottom=False))

    def _render_bottom_triangles(self):
        """
        Draw the 12 triangles at the bottom of the board (points 1 to 12) with alternating colors.
        """

        for i in range(BoardParameters.triangle_count_per_side):
            if i < BoardParameters.triangle_count_per_side // 2:
                # Bottom right quadrant: corresponds to points 6 to 1 (reverse order)
                x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                     BoardParameters.bar_width + (5 - i) * BoardParameters.triangle_width)
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                x = BoardParameters.margin + (11 - i) * BoardParameters.triangle_width
            y = BoardParameters.margin + BoardParameters.board_height
            self.drawer.draw_triangle(Colors.triangle_colors[i % 2],
                                      self._get_points_from_coordinates(x, y, is_bottom=True))

    @staticmethod
    def _get_points_from_coordinates(x: float, y: float, is_bottom: bool) -> list:
        """
        Get the 3 points defining a triangle based on the provided coordinates for its left corner.
        The fact whether a triangle is at the bottom or top of the board is also needed, to determine the
        position of the top vertex (lower pygame pixel coordinate for triangles at the bottom) 
 
        :param x: x coordinate of the left corner of the base
        :param y: y coordinate of the left corner of the base
        :param is_bottom: flag whether the triangle is at the bottom of the board (True if yes)
        :return: the 3 points defining a triangle on the pygame screen
        """

        return [
            (x, y),  # left corner of the base
            (x + BoardParameters.triangle_width, y),  # right corner of the base
            (x + BoardParameters.triangle_width / 2, y + ((-1) ** is_bottom) * BoardParameters.triangle_height)  # apex
        ]

    @staticmethod
    def _get_checker_color_from_position_value(position_value: int) -> tuple:
        """
        Helper function of the renderer to get the checker color based on the value of the position.
        As we know that at no position can there be checkers of both players, then a positive position value implies
        that there are only player1 checkers, while a negative, player2 checkers.

        :param position_value: value at the position (one of the 24)
        :return: the color of the checker
        """

        if position_value > 0:
            return Colors.player1_checker_color
        else:
            return Colors.player2_checker_color

    def _draw_stack(self, center_x: float, start_y: float, count: int, available_space: float,
                   orientation: Orientations, color: tuple):
        """
        Draw the specified amount of checkers in a given space.
        
        :param center_x: x coordinate of the center of the checkers
        :param start_y: y coordinate of the end of the first checker
        :param count: number of checkers to be drawn
        :param available_space: the space available to draw the checkers (in pygame pixels)
        :param orientation: enumeration whether the checkers grow from the top down (TOP) or bottom up (BOTTOM)
        :param color: color of the checkers to be drawn
        """

        max_fit = math.floor(available_space / (BoardParameters.checker_diameter + BoardParameters.spacing))
        if 0 < max_fit < count:
            # Draw a single checker with the count rendered on it.
            if orientation == Orientations.TOP:
                center_y = start_y + BoardParameters.checker_radius
            else:  # bottom
                center_y = start_y - BoardParameters.checker_radius

            self.drawer.draw_checker(color, (int(center_x), int(center_y)))
            self.text_handler.render_checker_text((int(center_x), int(center_y)), f"{count}")
            
        else:
            for j in range(count):
                if orientation == Orientations.TOP:
                    center_y = (start_y + j * (BoardParameters.checker_diameter + BoardParameters.spacing) +
                                BoardParameters.checker_radius)
                else:
                    center_y = (start_y - j * (BoardParameters.checker_diameter + BoardParameters.spacing) -
                                BoardParameters.checker_radius)
                self.drawer.draw_checker(color, (int(center_x), int(center_y)))

    def _render_checkers_in_top_triangles(self, positions: list):
        """
        Draw the checkers in the bottom triangles (points 13 to 24).

        :param positions: a list with the checkers in the 24 board positions
        """

        for idx in range(BoardParameters.triangle_count_per_side, BoardParameters.triangle_counts):
            position_value = positions[idx]
            count = abs(position_value)
            if count == 0:
                continue

            checker_color = self._get_checker_color_from_position_value(position_value)
            i = idx - 12
            if i < QUARTER_BOARD_SIZE:
                # Left quadrant (points 13-18)
                center_x = (BoardParameters.margin + i * BoardParameters.triangle_width +
                            BoardParameters.triangle_width / 2)
            else:
                # Right quadrant (points 19-24)
                center_x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + (i - 6) * BoardParameters.triangle_width +
                            BoardParameters.triangle_width / 2)

            # For top triangles, pieces are stacked downward from near the base (the top edge).
            start_y = BoardParameters.margin + BoardParameters.top_base_offset
            self._draw_stack(center_x, start_y, count,
                            BoardParameters.top_available, Orientations.TOP, checker_color)

    def _render_checkers_in_bottom_triangles(self, positions: list):
        """
        Draw the checkers in the bottom triangles (points 1 to 12).

        :param positions: a list with the checkers in the 24 board positions
        """

        for idx in range(BoardParameters.triangle_count_per_side):
            position_value = positions[idx]
            count = abs(position_value)
            if count == 0:
                continue

            checker_color = self._get_checker_color_from_position_value(position_value)
            if idx < 6:
                # Bottom right quadrant: points 6 to 1 (reverse order)
                i = 5 - idx
                center_x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2)
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                i = 11 - idx
                center_x = BoardParameters.margin + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2

            # For bottom triangles, pieces are stacked upward from near the base (the bottom edge).
            start_y = BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset
            self._draw_stack(center_x, start_y, count, BoardParameters.bottom_available,
                            Orientations.BOTTOM, checker_color)

    def _render_checkers_in_bar(self, bar: list):
        """
        Draw the checkers in the middle bar of the board. Checkers are drawn for both players,
        with player1 at the bottom and player2 at the top.

        :param bar: a list with the number of checkers in the bar for player1 and player2
        """

        # Top bar (black pieces):
        top_bar_available = BoardParameters.board_height / 2 - BoardParameters.top_base_offset
        bar_center_x = BoardParameters.margin + BoardParameters.board_width / 2

        self._draw_stack(bar_center_x, BoardParameters.margin + BoardParameters.top_base_offset, bar[1],
                        top_bar_available, Orientations.TOP, Colors.player2_checker_color)

        # Bottom bar (white pieces):
        bottom_bar_available = BoardParameters.board_height / 2 - BoardParameters.bottom_base_offset
        self._draw_stack(bar_center_x,
                        BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset,
                        bar[0], bottom_bar_available, Orientations.BOTTOM, Colors.player1_checker_color)

    def _render_borne_off_checkers(self, off: list):
        """
        Draw the borne-off checkers in the left section of the board. Checkers are drawn for both players,
        with player1 at the bottom and player2 at the top.

        :param off: a list with the number of borne-off checkers for player1 and player2
        """

        # For black off checkers (drawer from the top of the off column downward)
        off_top_available = BoardParameters.board_height / 2 - BoardParameters.top_base_offset
        off_center_x = BoardParameters.margin + BoardParameters.board_width + BoardParameters.off_width / 2

        self._draw_stack(off_center_x, BoardParameters.margin + BoardParameters.top_base_offset, off[1],
                        off_top_available, Orientations.TOP, Colors.player2_checker_color)

        # For white off checkers (drawer from the bottom of the off column upward)
        off_bottom_available = BoardParameters.board_height / 2 - BoardParameters.bottom_base_offset
        self._draw_stack(off_center_x,
                        BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset,
                        off[0], off_bottom_available, Orientations.BOTTOM, Colors.player1_checker_color)
