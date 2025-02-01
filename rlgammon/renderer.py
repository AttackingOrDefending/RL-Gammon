import pygame
import sys
import math


class BackgammonRenderer:
    def __init__(self, width=900, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Backgammon")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (34, 139, 34)  # dark green background
        self.triangle_color1 = (222, 184, 135)  # light brown
        self.triangle_color2 = (139, 69, 19)  # dark brown
        self.bar_color = (105, 105, 105)  # gray for the bar
        self.white_color = (255, 255, 255)
        self.black_color = (0, 0, 0)
        self.outline_color = (0, 0, 0)

        # Board geometry parameters:
        self.margin = 50
        self.board_width = 700
        self.board_height = 500
        self.bar_width = 50
        # Off-board column width:
        self.off_width = 60

        # The triangle (point) areas:
        self.triangle_height = (self.board_height // 2) - 20
        # There are 12 triangles per half.
        self.triangle_width = (self.board_width - self.bar_width) / 12

        # Prepare a font for drawing numbers on overloaded stacks.
        self.font = pygame.font.SysFont(None, 20)

    def draw_stack(self, center_x, start_y, count, piece_radius, available_space, orientation, color):
        """
        Draw a vertical stack of checkers at the given x, starting from start_y.
        If count is too high to fit within available_space using the standard spacing,
        then draw only one checker with the count drawn on it.

        :param center_x: x-coordinate for the centers of the checkers.
        :param start_y: starting y coordinate (for top triangles, this is from the top; for bottom triangles, from the bottom).
        :param count: number of checkers to draw.
        :param piece_radius: radius of a checker.
        :param available_space: vertical space available for stacking.
        :param orientation: "top" means stacking downward; "bottom" means stacking upward.
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
            pygame.draw.circle(self.screen, self.outline_color, (int(center_x), int(center_y)), piece_radius, 1)
            text = self.font.render(f"{count}", True, self.outline_color)
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)
        else:
            for j in range(count):
                if orientation == "top":
                    center_y = start_y + j * (piece_diameter + spacing) + piece_radius
                else:
                    center_y = start_y - j * (piece_diameter + spacing) - piece_radius
                pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), piece_radius)
                pygame.draw.circle(self.screen, self.outline_color, (int(center_x), int(center_y)), piece_radius, 1)

    def render(self, positions, bar, off, wait=True):
        """
        Render the backgammon board.

        :param positions: List of 24 integers. For each point:
                          positive → white pieces, negative → black pieces.
                          positions[0] is point 1; positions[23] is point 24.
        :param bar: List of 2 integers: [white_on_bar, black_on_bar].
                      (By convention, black pieces on the bar are drawn on the top half,
                       white pieces on the bottom half.)
        :param off: List of 2 integers: [white_off, black_off] to be drawn in an extra column.
        :param wait: If True, waits for a key press or window close.
        """
        if len(positions) != 24:
            raise ValueError("positions must be a list of length 24")
        if len(bar) != 2:
            raise ValueError("bar must be a list of length 2")
        if len(off) != 2:
            raise ValueError("off must be a list of length 2")

        # Clear the screen.
        self.screen.fill(self.bg_color)

        # Draw the board playing area.
        board_rect = pygame.Rect(self.margin, self.margin, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, self.bg_color, board_rect)
        pygame.draw.rect(self.screen, self.outline_color, board_rect, 3)

        # Draw the central bar.
        bar_x = self.margin + (self.board_width - self.bar_width) / 2
        bar_rect = pygame.Rect(bar_x, self.margin, self.bar_width, self.board_height)
        pygame.draw.rect(self.screen, self.bar_color, bar_rect)
        pygame.draw.rect(self.screen, self.outline_color, bar_rect, 2)

        # Draw the off-board column to the right of the board.
        off_rect = pygame.Rect(self.margin + self.board_width, self.margin, self.off_width, self.board_height)
        pygame.draw.rect(self.screen, self.bg_color, off_rect)
        pygame.draw.rect(self.screen, self.outline_color, off_rect, 3)

        # --- Draw the triangles for the points ---
        triangle_colors = [self.triangle_color1, self.triangle_color2]

        # Top half: points 13-24 (drawn left-to-right)
        for i in range(12):
            if i < 6:
                # Left quadrant (points 13-18)
                x = self.margin + i * self.triangle_width
            else:
                # Right quadrant (points 19-24)
                x = self.margin + ((self.board_width - self.bar_width) / 2) + self.bar_width + (
                            i - 6) * self.triangle_width
            y = self.margin
            points = [
                (x, y),  # left corner of the base
                (x + self.triangle_width, y),  # right corner of the base
                (x + self.triangle_width / 2, y + self.triangle_height)  # apex
            ]
            color = triangle_colors[i % 2]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, self.outline_color, points, 1)

        # Bottom half: points 1-12 (drawn right-to-left in each quadrant so they mirror the top)
        for i in range(12):
            if i < 6:
                # Bottom right quadrant: corresponds to points 6 to 1 (reverse order)
                x = self.margin + ((self.board_width - self.bar_width) / 2) + self.bar_width + (
                            5 - i) * self.triangle_width
            else:
                # Bottom left quadrant: corresponds to points 12 to 7 (reverse order)
                x = self.margin + (11 - i) * self.triangle_width
            y = self.margin + self.board_height
            points = [
                (x, y),  # left corner of the base (bottom)
                (x + self.triangle_width, y),  # right corner
                (x + self.triangle_width / 2, y - self.triangle_height)  # apex (pointing upward)
            ]
            color = triangle_colors[i % 2]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, self.outline_color, points, 1)

        # --- Draw the checkers on the board ---
        piece_radius = int(self.triangle_width * 0.4)
        piece_diameter = piece_radius * 2

        # Move the men closer to the base: set the starting offsets to be exactly one piece_radius away from the board edge.
        top_base_offset = piece_radius / 2  # for top triangles, the first checker is drawn at margin + piece_radius
        bottom_base_offset = piece_radius / 2  # for bottom triangles, drawn at margin+board_height - piece_radius

        # Available vertical space in a triangle:
        top_available = self.triangle_height - top_base_offset
        bottom_available = self.triangle_height - bottom_base_offset

        # Top half: points 13-24 (indices 12 to 23)
        for idx in range(12, 24):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            # For top triangles, positive means white, negative means black.
            piece_color = self.white_color if positions[idx] > 0 else self.black_color

            i = idx - 12
            if i < 6:
                # Left quadrant (points 13-18)
                center_x = self.margin + i * self.triangle_width + self.triangle_width / 2
            else:
                # Right quadrant (points 19-24)
                center_x = self.margin + ((self.board_width - self.bar_width) / 2) + self.bar_width + (
                            i - 6) * self.triangle_width + self.triangle_width / 2

            # For top triangles, pieces are stacked downward from near the base (the top edge).
            start_y = self.margin + top_base_offset
            self.draw_stack(center_x, start_y, count, piece_radius, top_available, "top", piece_color)

        # Bottom half: points 1-12 (indices 0 to 11)
        for idx in range(0, 12):
            count = abs(positions[idx])
            if positions[idx] == 0:
                continue
            piece_color = self.white_color if positions[idx] > 0 else self.black_color

            if idx < 6:
                # Bottom right quadrant: points 6 to 1 (reverse order)
                i = 5 - idx
                center_x = self.margin + ((
                                                      self.board_width - self.bar_width) / 2) + self.bar_width + i * self.triangle_width + self.triangle_width / 2
            else:
                # Bottom left quadrant: points 12 to 7 (reverse order)
                i = 11 - idx
                center_x = self.margin + i * self.triangle_width + self.triangle_width / 2

            # For bottom triangles, pieces are stacked upward from near the base (the bottom edge).
            start_y = self.margin + self.board_height - bottom_base_offset
            self.draw_stack(center_x, start_y, count, piece_radius, bottom_available, "bottom", piece_color)

        # --- Draw the checkers on the bar ---
        bar_center_x = self.margin + self.board_width / 2
        # Top bar (black pieces):
        top_bar_available = self.board_height / 2 - top_base_offset
        self.draw_stack(bar_center_x, self.margin + top_base_offset, bar[1], piece_radius, top_bar_available, "top", self.black_color)
        # Bottom bar (white pieces):
        bottom_bar_available = self.board_height / 2 - bottom_base_offset
        self.draw_stack(bar_center_x, self.margin + self.board_height - bottom_base_offset, bar[0], piece_radius,
                        bottom_bar_available, "bottom", self.white_color)

        # --- Draw the off-board column (beared off pieces) ---
        off_center_x = self.margin + self.board_width + self.off_width / 2
        # For black off checkers (draw from the top of the off column downward)
        off_top_available = self.board_height / 2 - top_base_offset
        self.draw_stack(off_center_x, self.margin + top_base_offset, off[1], piece_radius, off_top_available, "top", self.black_color)
        # For white off checkers (draw from the bottom of the off column upward)
        off_bottom_available = self.board_height / 2 - bottom_base_offset
        self.draw_stack(off_center_x, self.margin + self.board_height - bottom_base_offset, off[0], piece_radius,
                        off_bottom_available, "bottom", self.white_color)

        pygame.display.flip()

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
    import backgammon
    bg = backgammon.Backgammon()
    bg.bar[0] = 8
    bg.bar[1] = 2
    bg.board[0] = 11
    bg.off[0] = 15
    bg.off[1] = 2
    renderer = BackgammonRenderer()
    renderer.render(bg.board, bg.bar, bg.off)
