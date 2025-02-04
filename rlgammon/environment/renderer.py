import pygame
import sys
import math
import time

from rlgammon.environment.render_data.colors import Colors
from rlgammon.environment.render_data.board_parameters import BoardParameters
from rlgammon.environment.drawer.drawer import Drawer


class BackgammonRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((BoardParameters.screen_width, BoardParameters.screen_height))
        pygame.display.set_caption("Backgammon")

        # Prepare a font for drawing numbers on overloaded stacks.
        self.font = pygame.font.SysFont(None, 20)

        # Init drawer class used for drawing objects on the pyme display
        self.drawer = Drawer(self.screen)

    def draw_stack(self, center_x, start_y, count, piece_radius, available_space, orientation, color):
        """
        Draw a vertical stack of checkers at the given x, starting from start_y.
        If count is too high to fit within available_space using the standard spacing,
        then drawer only one checker with the count drawn on it.

        :param center_x: x-coordinate for the centers of the checkers.
        :param start_y: starting y coordinate (for top triangles, this is from the top; for bottom triangles, from the bottom).
        :param count: number of checkers to drawer.
        :param piece_radius: radius of a checker.
        :param available_space: vertical space available for stacking.
        :param orientation: "top" means stacking downward; "bottom" means stacking upward.
        :param color: color of the checker.
        """

        spacing = 1  # reduced spacing so more checkers can fit
        piece_diameter = piece_radius * 2
        max_fit = math.floor(available_space / (piece_diameter + spacing))
        if count > max_fit and max_fit > 0:
            # Draw a single checker with the count rendered on it.
            if orientation == "top":
                center_y = start_y + piece_radius
            else:  # bottom
                center_y = start_y - piece_radius
            # Use the appropriate color for the count marker:
            # (Here we simply use white if count is positive, black if negative; in our calls we already pass abs(count))
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

    def render(self, positions, bar, off, render_duration_in_s=2):
        if len(positions) != 24:
            raise ValueError("positions must be a list of length 24")
        if len(bar) != 2:
            raise ValueError("bar must be a list of length 2")
        if len(off) != 2:
            raise ValueError("off must be a list of length 2")

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
        self.render_top_triangles()

        # Bottom half: points 1-12 (drawn right-to-left in each quadrant so they mirror the top)
        self.render_bottom_triangles()

        # Top half: points 13-24 (indices 12 to 23)
        for idx in range(12, 24):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            # For top triangles, positive means white, negative means black.
            piece_color = Colors.player1_checker_color if positions[idx] > 0 else Colors.player2_checker_color

            i = idx - 12
            if i < 6:
                # Left quadrant (points 13-18)
                center_x = BoardParameters.margin + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2
            else:
                # Right quadrant (points 19-24)
                center_x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + (i - 6) * BoardParameters.triangle_width + BoardParameters.triangle_width / 2)

            # For top triangles, pieces are stacked downward from near the base (the top edge).
            start_y = BoardParameters.margin + BoardParameters.top_base_offset
            self.draw_stack(center_x, start_y, count, BoardParameters.checker_radius, BoardParameters.top_available, "top", piece_color)

        # Bottom half: points 1-12 (indices 0 to 11)
        for idx in range(0, 12):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            piece_color = Colors.player1_checker_color if positions[idx] > 0 else Colors.player2_checker_color

            if idx < 6:
                # Bottom right quadrant: points 6 to 1 (reverse order)
                i = 5 - idx
                center_x = (BoardParameters.margin + ((
                                                      BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                            BoardParameters.bar_width + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2)
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                i = 11 - idx
                center_x = BoardParameters.margin + i * BoardParameters.triangle_width + BoardParameters.triangle_width / 2

            # For bottom triangles, pieces are stacked upward from near the base (the bottom edge).
            start_y = BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset
            self.draw_stack(center_x, start_y, count, BoardParameters.checker_radius, BoardParameters.bottom_available, "bottom", piece_color)

        # --- Draw the checkers on the bar ---
        bar_center_x = BoardParameters.margin + BoardParameters.board_width / 2
        # Top bar (black pieces):
        top_bar_available = BoardParameters.board_height / 2 - BoardParameters.top_base_offset
        self.draw_stack(bar_center_x, BoardParameters.margin + BoardParameters.top_base_offset, bar[1], BoardParameters.checker_radius,
                        top_bar_available, "top", Colors.player2_checker_color)
        # Bottom bar (white pieces):
        bottom_bar_available = BoardParameters.board_height / 2 - BoardParameters.bottom_base_offset
        self.draw_stack(bar_center_x, BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset, bar[0],
                        BoardParameters.checker_radius, bottom_bar_available, "bottom", Colors.player1_checker_color)

        # --- Draw the off-board column (beared off pieces) ---
        off_center_x = BoardParameters.margin + BoardParameters.board_width + BoardParameters.off_width / 2
        # For black off checkers (drawer from the top of the off column downward)
        off_top_available = BoardParameters.board_height / 2 - BoardParameters.top_base_offset
        self.draw_stack(off_center_x, BoardParameters.margin + BoardParameters.top_base_offset, off[1], BoardParameters.checker_radius,
                        off_top_available,"top", Colors.player2_checker_color)
        # For white off checkers (drawer from the bottom of the off column upward)
        off_bottom_available = BoardParameters.board_height / 2 - BoardParameters.bottom_base_offset
        self.draw_stack(off_center_x, BoardParameters.margin + BoardParameters.board_height - BoardParameters.bottom_base_offset, off[0],
                        BoardParameters.checker_radius, off_bottom_available, "bottom", Colors.player1_checker_color)


        start_time = time.time()
        while time.time() - start_time < render_duration_in_s:
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    return

    def render_top_triangles(self):
        """
        TODO

        :return:
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
                                      self.get_point_from_coordinates(x, y, is_bottom=False))

    def render_bottom_triangles(self):
        """
        TODO

        :return:
        """

        for i in range(BoardParameters.triangle_count_per_side):
            if i < BoardParameters.triangle_count_per_side // 2:
                # Bottom right quadrant: corresponds to points 6 to 1 (reverse order)
                x = (BoardParameters.margin + ((BoardParameters.board_width - BoardParameters.bar_width) / 2) +
                     BoardParameters.bar_width + (5 - i) * BoardParameters.triangle_width)
            else:
                # Bottom left quadrant: corresponds to points 12 to 7 (reverse order)
                x = BoardParameters.margin + (11 - i) * BoardParameters.triangle_width
            y = BoardParameters.margin + BoardParameters.board_height
            self.drawer.draw_triangle(Colors.triangle_colors[i % 2],
                                      self.get_point_from_coordinates(x, y, is_bottom=True))


    @staticmethod
    def get_point_from_coordinates(x: float, y: float, is_bottom: bool):
        """
        TODO

        :param x:
        :param y:
        :param is_bottom:
        :return:
        """

        return [
            (x, y),  # left corner of the base
            (x + BoardParameters.triangle_width, y),  # right corner of the base
            (x + BoardParameters.triangle_width / 2, y + ((-1) ** is_bottom) * BoardParameters.triangle_height)  # apex
        ]

    def render_checkers_in_top_triangles(self):
        pass

    def render_checkers_in_bottom_triangles(self):
        pass

    def render_checkers_in_bar(self):
        pass

if __name__ == "__main__":
    from rlgammon.environment import backgammon
    bg = backgammon.Backgammon()
    bg.bar[0] = 8
    bg.bar[1] = 2
    bg.board[0] = 11
    bg.off[0] = 15
    bg.off[1] = 2
    renderer = BackgammonRenderer()
    renderer.render(bg.board, bg.bar, bg.off)
