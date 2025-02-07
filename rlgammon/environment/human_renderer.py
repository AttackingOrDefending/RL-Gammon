"""Module for rendering the backgammon board using pygame."""

import math
import sys

import numpy as np
import numpy.typing as npt
import pygame

from rlgammon.environment.render_data.board_parameters import BoardParameters
from rlgammon.environment.render_data.colors import Colors

QUARTER_BOARD_SIZE = 6  # Number of points in each quarter of the board


def get_x_top(i: int) -> float:
    """
    Calculate the x-coordinate for the top triangles.

    :param i: index of the triangle (0-11)
    :return: x-coordinate for the triangle's position

    Variables:
    - x: The calculated x-coordinate for the triangle
    """
    x: float
    if i < QUARTER_BOARD_SIZE:
        # Left quadrant (points 13-18)
        x = BoardParameters.margin + i * BoardParameters.triangle_width
    else:
        # Right quadrant (points 19-24)
        x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
             BoardParameters.bar_width + (i - 6) * BoardParameters.triangle_width)
    return x


class BackgammonRenderer:
    """Class for rendering the backgammon board using pygame."""

    def __init__(self) -> None:
        """
        Initialize the backgammon renderer.

        Variables:
        - screen: Pygame display surface for rendering
        - clock: Pygame clock for controlling frame rate
        - font: Pygame font for rendering text
        """
        pygame.init()
        self.screen = pygame.display.set_mode((BoardParameters.screen_width, BoardParameters.screen_height))
        pygame.display.set_caption("Backgammon")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

    def draw_stack(self, center_x: float, start_y: float, count: int, piece_radius: int, available_space: float,
                   orientation: str, color: tuple[int, int, int]) -> None:
        """
        Draw a vertical stack of checkers at the given x, starting from start_y.

        :param center_x: x-coordinate for the centers of the checkers
        :param start_y: starting y coordinate (top for top triangles, bottom for bottom triangles)
        :param count: number of checkers to draw
        :param piece_radius: radius of a checker
        :param available_space: vertical space available for stacking
        :param orientation: "top" for stacking downward, "bottom" for stacking upward
        :param color: RGB color tuple for the checker

        Variables:
        - spacing: Space between checkers in the stack
        - piece_diameter: Full diameter of a checker piece
        - max_fit: Maximum number of checkers that can fit in the available space
        - center_y: y-coordinate for the center of each checker
        - text: Rendered text surface for checker count
        - text_rect: Rectangle for positioning the rendered text
        """
        spacing = 1  # reduced spacing so more checkers can fit
        piece_diameter = piece_radius * 2
        max_fit = math.floor(available_space / (piece_diameter + spacing))
        if count > max_fit > 0:
            center_y = start_y + piece_radius if orientation == "top" else start_y - piece_radius
            pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), piece_radius)
            pygame.draw.circle(self.screen, Colors.outline_color,
                               (int(center_x), int(center_y)), piece_radius, 1)
            text = self.font.render(f"{count}", True, Colors.outline_color)
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)
        else:
            for j in range(count):
                if orientation == "top":
                    center_y = start_y + j * (piece_diameter + spacing) + piece_radius
                else:
                    center_y = start_y - j * (piece_diameter + spacing) - piece_radius
                pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), piece_radius)
                pygame.draw.circle(self.screen, Colors.outline_color,
                                   (int(center_x), int(center_y)), piece_radius, 1)

    def render(self, positions: npt.NDArray[np.int8], bar: npt.NDArray[np.int8], off: npt.NDArray[np.int8],
               wait: bool = True) -> None:
        """
        Render the backgammon board.

        :param positions: 24-integer array for point positions (positive=white, negative=black)
        :param bar: 2-integer array [white_on_bar, black_on_bar]
        :param off: 2-integer array [white_off, black_off] for off-board pieces
        :param wait: If True, waits for key press or window close

        Variables:
        - board_rect: Rectangle defining the main board area
        - bar_x: x-coordinate for the central bar
        - bar_rect: Rectangle defining the central bar area
        - off_rect: Rectangle defining the off-board area
        - triangle_colors: List of alternating colors for board triangles
        - piece_radius: Radius of checker pieces
        - top_base_offset: Offset from top edge for piece placement
        - bottom_base_offset: Offset from bottom edge for piece placement
        - top_available: Available vertical space in top triangles
        - bottom_available: Available vertical space in bottom triangles
        - points: List of coordinates defining triangle vertices
        - x: x-coordinate for triangle positioning
        - y: y-coordinate for triangle positioning
        - color: Current triangle color
        - count: Number of pieces on a point
        - piece_color: Color for current piece being drawn
        - center_x: x-coordinate for piece center
        - start_y: Starting y-coordinate for piece stack
        - i: Loop counter for triangle/point positions
        - idx: Index for accessing board positions array
        - bar_center_x: x-coordinate for bar piece centers
        - top_bar_available: Available space for pieces in top bar
        - bottom_bar_available: Available space for pieces in bottom bar
        - off_center_x: x-coordinate for off-board piece centers
        - off_top_available: Available space for off-board pieces at top
        - off_bottom_available: Available space for off-board pieces at bottom
        """
        # Clear the screen
        self.screen.fill(Colors.bg_color)

        # Draw the board playing area
        board_rect = pygame.Rect(BoardParameters.margin, BoardParameters.margin,
                                 BoardParameters.board_width, BoardParameters.board_height)
        pygame.draw.rect(self.screen, Colors.bg_color, board_rect)
        pygame.draw.rect(self.screen, Colors.outline_color, board_rect, 3)

        # Draw the central bar
        bar_x = BoardParameters.margin + (BoardParameters.board_width - BoardParameters.bar_width) / 2
        bar_rect = pygame.Rect(bar_x, BoardParameters.margin,
                               BoardParameters.bar_width, BoardParameters.board_height)
        pygame.draw.rect(self.screen, Colors.bar_color, bar_rect)
        pygame.draw.rect(self.screen, Colors.outline_color, bar_rect, 2)

        # Draw the off-board column
        off_rect = pygame.Rect(BoardParameters.margin + BoardParameters.board_width,
                               BoardParameters.margin, BoardParameters.off_width, BoardParameters.board_height)
        pygame.draw.rect(self.screen, Colors.bg_color, off_rect)
        pygame.draw.rect(self.screen, Colors.outline_color, off_rect, 3)

        # Draw the triangles for the points
        triangle_colors = [Colors.triangle_color1, Colors.triangle_color2]

        # Top half: points 13-24 (drawn left-to-right)
        x: float
        for i in range(12):
            x = get_x_top(i)
            y = BoardParameters.margin
            points = [
                (x, y),  # left corner of the base
                (x + BoardParameters.triangle_width, y),  # right corner of the base
                (x + BoardParameters.triangle_width / 2, y + BoardParameters.triangle_height),  # apex
            ]
            color = triangle_colors[i % 2]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, Colors.outline_color, points, 1)

        # Bottom half: points 1-12 (drawn right-to-left in each quadrant)
        for i in range(12):
            if i < QUARTER_BOARD_SIZE:
                # Bottom right quadrant: points 6 to 1 (reverse order)
                x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                     BoardParameters.bar_width + (5 - i) * BoardParameters.triangle_width)
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                x = BoardParameters.margin + (11 - i) * BoardParameters.triangle_width
            y = BoardParameters.margin + BoardParameters.board_height
            points = [
                (x, y),  # left corner of the base (bottom)
                (x + BoardParameters.triangle_width, y),  # right corner
                (x + BoardParameters.triangle_width / 2, y - BoardParameters.triangle_height),  # apex (pointing upward)
            ]
            color = triangle_colors[i % 2]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, Colors.outline_color, points, 1)

        # Draw the checkers
        piece_radius = int(BoardParameters.triangle_width * 0.4)
        top_base_offset = piece_radius / 2
        bottom_base_offset = piece_radius / 2
        top_available = BoardParameters.triangle_height - top_base_offset
        bottom_available = BoardParameters.triangle_height - bottom_base_offset

        # Top half: points 13-24 (indices 12 to 23)
        for idx in range(12, 24):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            piece_color = Colors.player1_checker_color if positions[idx] > 0 else Colors.player2_checker_color

            i = idx - 12
            if i < QUARTER_BOARD_SIZE:
                # Left quadrant (points 13-18)
                center_x = BoardParameters.margin + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2
            else:
                # Right quadrant (points 19-24)
                center_x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + (i - 6) * BoardParameters.triangle_width +
                            BoardParameters.triangle_width / 2)

            start_y = BoardParameters.margin + top_base_offset
            self.draw_stack(center_x, start_y, int(count), piece_radius, top_available, "top", piece_color)

        # Bottom half: points 1-12 (indices 0 to 11)
        for idx in range(12):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            piece_color = Colors.player1_checker_color if positions[idx] > 0 else Colors.player2_checker_color

            if idx < QUARTER_BOARD_SIZE:
                # Bottom right quadrant: points 6 to 1 (reverse order)
                i = 5 - idx
                center_x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + i * BoardParameters.triangle_width +
                            BoardParameters.triangle_width / 2)
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                i = 11 - idx
                center_x = BoardParameters.margin + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2

            start_y = BoardParameters.margin + BoardParameters.board_height - bottom_base_offset
            self.draw_stack(center_x, start_y, int(count), piece_radius, bottom_available, "bottom", piece_color)

        # Draw the bar pieces
        bar_center_x = BoardParameters.margin + BoardParameters.board_width / 2
        top_bar_available = BoardParameters.board_height / 2 - top_base_offset
        self.draw_stack(bar_center_x, BoardParameters.margin + top_base_offset, int(bar[1]), piece_radius,
                        top_bar_available, "top", Colors.player2_checker_color)
        bottom_bar_available = BoardParameters.board_height / 2 - bottom_base_offset
        self.draw_stack(bar_center_x, BoardParameters.margin + BoardParameters.board_height - bottom_base_offset,
                        int(bar[0]),
                        piece_radius, bottom_bar_available, "bottom", Colors.player1_checker_color)

        # Draw the off-board pieces
        off_center_x = BoardParameters.margin + BoardParameters.board_width + BoardParameters.off_width / 2
        off_top_available = BoardParameters.board_height / 2 - top_base_offset
        self.draw_stack(off_center_x, BoardParameters.margin + top_base_offset, int(off[1]), piece_radius,
                        off_top_available, "top", Colors.player2_checker_color)
        off_bottom_available = BoardParameters.board_height / 2 - bottom_base_offset
        self.draw_stack(off_center_x, BoardParameters.margin + BoardParameters.board_height - bottom_base_offset,
                        int(off[0]),
                        piece_radius, off_bottom_available, "bottom", Colors.player1_checker_color)

        pygame.display.flip()

        self.wait(wait)

    def wait(self, wait: bool) -> None:
        """
        Sleep until a key press or window close event.

        :param wait: If True, enters wait loop for events

        Variables:
        - event: Pygame event from the event queue
        """
        if wait:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        return
                self.clock.tick(30)


if __name__ == "__main__":
    # Test code for rendering a sample board state
    from rlgammon.environment import backgammon
    bg = backgammon.Backgammon()
    bg.bar[0] = 8  # White pieces on bar
    bg.bar[1] = 2  # Black pieces on bar
    bg.board[0] = 11  # Pieces on first point
    bg.off[0] = 15  # White pieces off
    bg.off[1] = 2  # Black pieces off
    renderer = BackgammonRenderer()
    renderer.render(bg.board, bg.bar, bg.off)
